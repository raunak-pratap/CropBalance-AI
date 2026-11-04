"""Read-only quality audit for the Dharashiv local-market archive."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_INPUT = Path("data/raw/agmarknet_weekly/dharashiv_prices_v3.csv")
DEFAULT_RUNS = Path("data/raw/agmarknet_weekly/collection_runs.csv")
DEFAULT_OUTPUT = Path("data/processed/dharashiv_mandi_audit")
from services.agmarknet_local_archive import parse_agmarknet_date
EXPECTED_MARKETS = {
    "APMC Bhoom": "1959", "APMC Dharashiv": "1655", "APMC Kalamb": "1599",
    "APMC Murum": "1600", "APMC Paranda": "1960", "APMC Tuljapur": "1961",
    "APMC Umarga": "1962", "APMC Washi": "2053",
}
EXPECTED_CROPS = ["wheat", "onion", "soybean", "tomato"]


def load_archive(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Archive not found: {path}. Run scripts/sync_dharashiv_mandi_archive.py first."
        )
    df = pd.read_csv(path, dtype={"market_id": "string", "commodity_id": "string"})
    date_col = "observation_date" if "observation_date" in df.columns else "date"
    price_col = "reported_price" if "reported_price" in df.columns else "reported_price_numeric"
    if date_col not in df.columns:
        raise ValueError("Archive must contain observation_date or date")
    if price_col not in df.columns:
        raise ValueError("Archive must contain reported_price")
    df["observation_date"] = df[date_col].map(parse_agmarknet_date)
    df["reported_price_numeric"] = pd.to_numeric(df[price_col], errors="coerce")
    had_price_status = "price_status" in df.columns
    for col in ["crop", "market", "variety", "price_unit", "price_status", "data_source"]:
        if col not in df.columns:
            df[col] = None
    if "arrivals_value" in df.columns:
        df["arrivals_numeric"] = pd.to_numeric(df["arrivals_value"], errors="coerce")
    elif "arrivals_tonnes" in df.columns:
        df["arrivals_numeric"] = pd.to_numeric(df["arrivals_tonnes"], errors="coerce")
    else:
        df["arrivals_numeric"] = np.nan
    if "endpoint" not in df.columns:
        df["endpoint"] = "legacy_archive_endpoint_unknown"
    if "price_metric" not in df.columns:
        df["price_metric"] = "unverified"
    if not had_price_status:
        df["price_status"] = np.where(df["reported_price_numeric"].gt(0), "legacy_reported_unverified", "legacy_missing_or_invalid")
    return df


def group_quality(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["crop", "market", "market_id"]
    for keys, group in df.groupby(group_cols, dropna=False):
        dates = pd.DatetimeIndex(group["observation_date"].dropna().unique()).sort_values()
        gaps = dates.to_series().diff().dt.days.dropna()
        valid = group["reported_price_numeric"].gt(0) & group["observation_date"].notna()
        rows.append({
            "crop": keys[0], "market": keys[1], "market_id": keys[2],
            "rows": int(len(group)),
            "positive_numeric_reported_rows": int(valid.sum()),
            "unique_observation_dates": int(len(dates)),
            "first_date": dates.min().date().isoformat() if len(dates) else None,
            "last_date": dates.max().date().isoformat() if len(dates) else None,
            "max_gap_between_reported_dates_days": int(gaps.max()) if len(gaps) else None,
            "distinct_varieties": int(group["variety"].nunique(dropna=True)),
            "units": ";".join(sorted(group["price_unit"].dropna().astype(str).unique())),
            "sources": ";".join(sorted(group["data_source"].dropna().astype(str).unique())),
            "price_metrics": ";".join(sorted(group["price_metric"].dropna().astype(str).unique())) if "price_metric" in group.columns else "unknown_unverified",
            "missing_calendar_dates_count": int(max((dates.max() - dates.min()).days + 1 - len(dates), 0)) if len(dates) else 0,
            "missing_calendar_dates_example": ";".join(pd.date_range(dates.min(), dates.max(), freq="D").difference(dates).strftime("%Y-%m-%d").tolist()[:20]) if len(dates) else "",
            "arrivals_rows": int(group["arrivals_numeric"].notna().sum()),
            "price_min": float(group.loc[valid, "reported_price_numeric"].min()) if valid.any() else None,
            "price_median": float(group.loc[valid, "reported_price_numeric"].median()) if valid.any() else None,
            "price_max": float(group.loc[valid, "reported_price_numeric"].max()) if valid.any() else None,
        })
    return pd.DataFrame(rows)


def duplicate_report(df: pd.DataFrame) -> pd.DataFrame:
    keys = [c for c in ["observation_date", "market_id", "commodity_id", "variety", "price_unit", "data_source", "endpoint"] if c in df.columns]
    if not keys:
        return pd.DataFrame()
    work = df.copy()
    work["price_value_for_conflict"] = work["reported_price_numeric"].astype("string")
    grouped = work.groupby(keys, dropna=False).agg(
        row_count=("reported_price_numeric", "size"),
        distinct_price_values=("price_value_for_conflict", "nunique"),
        values=("price_value_for_conflict", lambda x: " | ".join(sorted(set(x.dropna().tolist())))),
    ).reset_index()
    return grouped[grouped["row_count"].gt(1) | grouped["distinct_price_values"].gt(1)]


def outlier_report(df: pd.DataFrame) -> pd.DataFrame:
    valid = df[df["reported_price_numeric"].gt(0) & df["observation_date"].notna()].copy()
    rows = []
    for (crop, market), group in valid.groupby(["crop", "market"], dropna=False):
        values = group["reported_price_numeric"]
        if len(values) < 4:
            continue
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr <= 0:
            continue
        flagged = group[(values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)]
        for _, record in flagged.iterrows():
            rows.append({
                "crop": crop, "market": market,
                "observation_date": record["observation_date"].date().isoformat(),
                "reported_price": float(record["reported_price_numeric"]),
                "q1": float(q1), "q3": float(q3),
                "flag": "IQR_outlier_review_only",
            })
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    df = load_archive(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    coverage = group_quality(df)
    # Include all configured crop/market combinations so zero-reporting
    # combinations are visible rather than disappearing from the audit.
    expected_grid = pd.DataFrame([
        {"crop": crop, "market": market, "market_id": market_id}
        for crop in EXPECTED_CROPS
        for market, market_id in EXPECTED_MARKETS.items()
    ])
    if not coverage.empty:
        coverage["market_id"] = coverage["market_id"].astype("string")
    expected_grid["market_id"] = expected_grid["market_id"].astype("string")
    coverage = expected_grid.merge(coverage, on=["crop", "market", "market_id"], how="outer")
    for count_col in ["rows", "positive_numeric_reported_rows", "unique_observation_dates", "distinct_varieties", "arrivals_rows"]:
        if count_col in coverage.columns:
            coverage[count_col] = coverage[count_col].fillna(0).astype(int)
    duplicates = duplicate_report(df)
    outliers = outlier_report(df)

    coverage.to_csv(args.output_dir / "market_crop_coverage.csv", index=False)
    duplicates.to_csv(args.output_dir / "duplicate_groups.csv", index=False)
    outliers.to_csv(args.output_dir / "price_outlier_flags.csv", index=False)

    valid_price = df["reported_price_numeric"].gt(0)
    summary = {
        "input": str(args.input),
        "rows": int(len(df)),
        "unique_crops": int(df["crop"].nunique(dropna=True)),
        "unique_markets": int(df["market_id"].nunique(dropna=True)),
        "positive_numeric_reported_rows": int(valid_price.sum()),
        "missing_or_invalid_price_rows": int((~valid_price).sum()),
        "invalid_date_rows": int(df["observation_date"].isna().sum()),
        "duplicate_or_conflicting_groups": int(len(duplicates)),
        "price_unit_values": sorted(df["price_unit"].dropna().astype(str).unique().tolist()),
        "source_values": sorted(df["data_source"].dropna().astype(str).unique().tolist()),
        "price_status_counts": {str(k): int(v) for k, v in df["price_status"].fillna("missing_status").value_counts().items()},
        "arrivals_rows": int(df["arrivals_numeric"].notna().sum()),
        "variety_values": sorted(df["variety"].dropna().astype(str).unique().tolist()),
        "outlier_flags_review_only": int(len(outliers)),
        "coverage_note": "Gaps are calendar gaps between reported observations; this does not infer mandi trading/closure days.",
        "price_semantics_note": "Weekly collector stores the reported value as unverified; do not rename it modal without an endpoint-specific schema confirmation.",
        "arrivals_note": "The weekly commodity-price response used by the collector does not supply arrivals; missing arrivals are not zero.",
    }
    if args.runs.exists():
        runs = pd.read_csv(args.runs)
        if "status" in runs.columns:
            summary["collection_run_status_counts"] = {str(k): int(v) for k, v in runs["status"].value_counts(dropna=False).items()}
            runs.groupby(["crop", "market", "status"], dropna=False).size().reset_index(name="requests").to_csv(
                args.output_dir / "collection_run_status.csv", index=False
            )
    else:
        summary["collection_run_log"] = "not_found; only observation archive can be audited"

    (args.output_dir / "quality_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n--- DHARASHIV MARKET DATA QUALITY AUDIT ---")
    print(f"Rows: {summary['rows']} | Crops: {summary['unique_crops']} | Markets: {summary['unique_markets']}")
    print(f"Positive numeric reported values: {summary['positive_numeric_reported_rows']}")
    print(f"Missing/invalid prices: {summary['missing_or_invalid_price_rows']}")
    print(f"Invalid dates: {summary['invalid_date_rows']}")
    print(f"Duplicate/conflicting groups: {summary['duplicate_or_conflicting_groups']}")
    print(f"Price units: {summary['price_unit_values']}")
    print(f"Arrivals values present: {summary['arrivals_rows']}")
    print(f"IQR outlier flags (review only): {summary['outlier_flags_review_only']}")
    if not coverage.empty:
        print("\nCoverage by crop and market:")
        print(coverage.to_string(index=False))
    print("\nSaved audit to:", args.output_dir)
    print("No records were removed or modified by this audit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
