"""Read-only quality audit for the Dharashiv local-market archive.

This script reads a local AGMARKNET observation archive and writes quality
reports. It never modifies, deletes, deduplicates, or rewrites source records.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.agmarknet_local_archive import parse_agmarknet_date  # noqa: E402


DEFAULT_INPUT = Path("data/raw/agmarknet_weekly/dharashiv_prices_v3.csv")
DEFAULT_RUNS = Path("data/raw/agmarknet_weekly/collection_runs.csv")
DEFAULT_OUTPUT = Path("data/processed/dharashiv_mandi_audit")

EXPECTED_MARKETS = {
    "APMC Bhoom": "1959",
    "APMC Dharashiv": "1655",
    "APMC Kalamb": "1599",
    "APMC Murum": "1600",
    "APMC Paranda": "1960",
    "APMC Tuljapur": "1961",
    "APMC Umarga": "1962",
    "APMC Washi": "2053",
}
EXPECTED_CROPS = ["wheat", "onion", "soybean", "tomato"]

METADATA_REPORT_COLUMNS = [
    "observation_date",
    "market_id",
    "commodity_id",
    "data_source",
    "endpoint",
    "market_labels",
    "crop_labels",
    "commodity_names",
    "row_count",
    "variety_labels",
    "variety_ids",
    "price_units",
    "reported_price_raw_values",
    "price_status_values",
    "numeric_price_values",
    "distinct_numeric_price_values",
    "raw_response_files",
    "metadata_flags",
    "review_status",
]


def _is_missing(value: Any) -> bool:
    """Return True for scalar pandas/Python missing values."""
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(result) if isinstance(result, (bool, np.bool_)) else False


def _distinct_values(series: pd.Series) -> list[str]:
    """Return sorted distinct non-empty string values, excluding nulls."""
    values: set[str] = set()
    for value in series:
        if _is_missing(value):
            continue
        text = str(value).strip()
        if text and text.lower() not in {"nan", "none", "<na>"}:
            values.add(text)
    return sorted(values, key=str.casefold)


def _has_mixed_missingness(series: pd.Series) -> bool:
    """Return True when a metadata field is present on some rows but not others."""
    present = 0
    missing = 0
    for value in series:
        if _is_missing(value) or not str(value).strip():
            missing += 1
        else:
            present += 1
    return present > 0 and missing > 0


def _joined_values(group: pd.DataFrame, column: str) -> str:
    """Join distinct values from a column that may be absent."""
    if column not in group.columns:
        return ""
    return "; ".join(_distinct_values(group[column]))


def load_archive(path: Path) -> pd.DataFrame:
    """Load an archive and standardize columns needed by the audit."""
    if not path.exists():
        raise FileNotFoundError(
            f"Archive not found: {path}. "
            "Run scripts/sync_dharashiv_mandi_archive.py first."
        )

    source_columns = set(pd.read_csv(path, nrows=0).columns)
    df = pd.read_csv(
        path,
        dtype={"market_id": "string", "commodity_id": "string"},
        low_memory=False,
    )

    had_price_status = "price_status" in source_columns
    date_col = "observation_date" if "observation_date" in df.columns else "date"
    if date_col not in df.columns:
        raise ValueError("Archive must contain 'observation_date' or 'date'.")

    if "reported_price" in df.columns:
        price_col = "reported_price"
    elif "reported_price_numeric" in df.columns:
        price_col = "reported_price_numeric"
    else:
        raise ValueError(
            "Archive must contain 'reported_price' or 'reported_price_numeric'."
        )

    df["observation_date"] = df[date_col].map(parse_agmarknet_date)
    df["reported_price_numeric"] = pd.to_numeric(
        df[price_col], errors="coerce"
    )

    # Ensure a stable schema across current and legacy archive files.
    optional_columns = [
        "crop",
        "market",
        "market_id",
        "commodity_id",
        "commodity_name",
        "variety",
        "variety_id",
        "price_unit",
        "price_status",
        "data_source",
        "endpoint",
        "raw_response_file",
        "reported_price_raw",
        "price_metric",
    ]
    for column in optional_columns:
        if column not in df.columns:
            df[column] = pd.NA

    if not had_price_status:
        df["price_status"] = np.where(
            df["reported_price_numeric"].gt(0),
            "legacy_reported_unverified",
            "legacy_missing_or_invalid",
        )

    if "arrivals_value" in df.columns:
        df["arrivals_numeric"] = pd.to_numeric(
            df["arrivals_value"], errors="coerce"
        )
    elif "arrivals_tonnes" in df.columns:
        df["arrivals_numeric"] = pd.to_numeric(
            df["arrivals_tonnes"], errors="coerce"
        )
    else:
        df["arrivals_numeric"] = np.nan

    if df["endpoint"].isna().all():
        df["endpoint"] = "legacy_archive_endpoint_unknown"
    else:
        df["endpoint"] = df["endpoint"].fillna(
            "legacy_archive_endpoint_unknown"
        )

    df["price_metric"] = df["price_metric"].fillna("unverified")
    return df


def group_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize observation coverage separately from reported-price coverage.

    Observation dates include every collected row, including NR/missing prices.
    Reported-price dates include only rows with positive numeric price values.
    Gap values are elapsed days between consecutive dates, not counts of
    missing calendar days; max_missing_calendar_days_* reports the latter.
    """
    rows: list[dict[str, Any]] = []
    group_cols = ["crop", "market", "market_id"]

    for keys, group in df.groupby(group_cols, dropna=False):
        observation_dates = pd.DatetimeIndex(
            group["observation_date"].dropna().unique()
        ).sort_values()

        valid = (
            group["reported_price_numeric"].gt(0)
            & group["observation_date"].notna()
        )
        reported_price_dates = pd.DatetimeIndex(
            group.loc[valid, "observation_date"].dropna().unique()
        ).sort_values()

        observation_gaps = (
            observation_dates.to_series().diff().dt.days.dropna()
        )
        reported_price_gaps = (
            reported_price_dates.to_series().diff().dt.days.dropna()
        )

        if len(observation_dates):
            first_observation_date = observation_dates.min().date().isoformat()
            last_observation_date = observation_dates.max().date().isoformat()
            missing_observation_dates = pd.date_range(
                observation_dates.min(), observation_dates.max(), freq="D"
            ).difference(observation_dates)
            missing_observation_date_count = int(
                max(
                    (observation_dates.max() - observation_dates.min()).days
                    + 1
                    - len(observation_dates),
                    0,
                )
            )
            missing_observation_date_examples = ";".join(
                missing_observation_dates.strftime("%Y-%m-%d").tolist()[:20]
            )
        else:
            first_observation_date = None
            last_observation_date = None
            missing_observation_date_count = 0
            missing_observation_date_examples = ""

        if len(reported_price_dates):
            first_reported_price_date = (
                reported_price_dates.min().date().isoformat()
            )
            last_reported_price_date = (
                reported_price_dates.max().date().isoformat()
            )
        else:
            first_reported_price_date = None
            last_reported_price_date = None

        max_observation_gap = (
            int(observation_gaps.max()) if len(observation_gaps) else None
        )
        max_reported_price_gap = (
            int(reported_price_gaps.max())
            if len(reported_price_gaps)
            else None
        )

        rows.append(
            {
                "crop": keys[0],
                "market": keys[1],
                "market_id": keys[2],
                "rows": int(len(group)),
                "positive_numeric_reported_rows": int(valid.sum()),
                "unique_observation_dates": int(len(observation_dates)),
                "first_date": first_observation_date,
                "last_date": last_observation_date,
                "max_gap_between_observation_dates_days": max_observation_gap,
                "missing_calendar_dates_count": missing_observation_date_count,
                "missing_calendar_dates_example": missing_observation_date_examples,
                "unique_reported_price_dates": int(len(reported_price_dates)),
                "first_reported_price_date": first_reported_price_date,
                "last_reported_price_date": last_reported_price_date,
                # Compatibility field: now correctly means gaps between
                # consecutive dates with positive numeric reported prices.
                "max_gap_between_reported_dates_days": max_reported_price_gap,
                "max_missing_calendar_days_between_reported_prices": (
                    max(max_reported_price_gap - 1, 0)
                    if max_reported_price_gap is not None
                    else None
                ),
                "distinct_varieties": int(
                    group["variety"].nunique(dropna=True)
                ),
                "units": ";".join(_distinct_values(group["price_unit"])),
                "sources": ";".join(_distinct_values(group["data_source"])),
                "price_metrics": ";".join(
                    _distinct_values(group["price_metric"])
                ),
                "arrivals_rows": int(group["arrivals_numeric"].notna().sum()),
                "price_min": (
                    float(group.loc[valid, "reported_price_numeric"].min())
                    if valid.any()
                    else None
                ),
                "price_median": (
                    float(group.loc[valid, "reported_price_numeric"].median())
                    if valid.any()
                    else None
                ),
                "price_max": (
                    float(group.loc[valid, "reported_price_numeric"].max())
                    if valid.any()
                    else None
                ),
            }
        )

    return pd.DataFrame(rows)

def duplicate_report(df: pd.DataFrame) -> pd.DataFrame:
    """Report repeated rows under the original variety/unit-aware key."""
    keys = [
        column
        for column in [
            "observation_date",
            "market_id",
            "commodity_id",
            "variety",
            "price_unit",
            "data_source",
            "endpoint",
        ]
        if column in df.columns
    ]
    if not keys:
        return pd.DataFrame()

    work = df.copy()
    work["price_value_for_conflict"] = work[
        "reported_price_numeric"
    ].astype("string")

    grouped = (
        work.groupby(keys, dropna=False)
        .agg(
            row_count=("reported_price_numeric", "size"),
            distinct_price_values=("price_value_for_conflict", "nunique"),
            values=(
                "price_value_for_conflict",
                lambda values: " | ".join(
                    sorted(set(values.dropna().tolist()))
                ),
            ),
        )
        .reset_index()
    )
    return grouped[
        grouped["row_count"].gt(1)
        | grouped["distinct_price_values"].gt(1)
    ].reset_index(drop=True)


def outlier_report(df: pd.DataFrame) -> pd.DataFrame:
    """Flag within-crop/market IQR outliers for review only."""
    valid = df[
        df["reported_price_numeric"].gt(0)
        & df["observation_date"].notna()
    ].copy()
    rows: list[dict[str, Any]] = []

    for (crop, market), group in valid.groupby(
        ["crop", "market"], dropna=False
    ):
        values = group["reported_price_numeric"]
        if len(values) < 4:
            continue

        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        if pd.isna(iqr) or iqr <= 0:
            continue

        flagged = group[
            (values < q1 - 1.5 * iqr)
            | (values > q3 + 1.5 * iqr)
        ]
        for _, record in flagged.iterrows():
            rows.append(
                {
                    "crop": crop,
                    "market": market,
                    "observation_date": record[
                        "observation_date"
                    ].date().isoformat(),
                    "reported_price": float(
                        record["reported_price_numeric"]
                    ),
                    "q1": float(q1),
                    "q3": float(q3),
                    "flag": "IQR_outlier_review_only",
                }
            )

    return pd.DataFrame(
        rows,
        columns=[
            "crop",
            "market",
            "observation_date",
            "reported_price",
            "q1",
            "q3",
            "flag",
        ],
    )


def metadata_inconsistency_report(df: pd.DataFrame) -> pd.DataFrame:
    """Detect metadata differences for the same date/market/commodity/source.

    Variety and price unit are deliberately excluded from the grouping key so
    this report can detect changes in those fields across API snapshots.
    This function is read-only and never merges or modifies observations.
    """
    group_keys = [
        "observation_date",
        "market_id",
        "commodity_id",
        "data_source",
        "endpoint",
    ]
    if not all(column in df.columns for column in group_keys):
        return pd.DataFrame(columns=METADATA_REPORT_COLUMNS)

    rows: list[dict[str, Any]] = []

    for identity, group in df.groupby(
        group_keys,
        dropna=False,
        sort=True,
    ):
        if not isinstance(identity, tuple):
            identity = (identity,)

        # Some legacy archives and focused unit-test fixtures omit optional
        # metadata columns. Treat an absent column as having no values rather
        # than raising KeyError.
        variety_values = (
            _distinct_values(group["variety"])
            if "variety" in group.columns else []
        )
        variety_id_values = (
            _distinct_values(group["variety_id"])
            if "variety_id" in group.columns else []
        )
        unit_values = (
            _distinct_values(group["price_unit"])
            if "price_unit" in group.columns else []
        )
        market_values = (
            _distinct_values(group["market"])
            if "market" in group.columns else []
        )
        crop_values = (
            _distinct_values(group["crop"])
            if "crop" in group.columns else []
        )
        commodity_name_values = (
            _distinct_values(group["commodity_name"])
            if "commodity_name" in group.columns else []
        )

        flags: list[str] = []
        if len(variety_values) > 1:
            flags.append("multiple_variety_labels")
        if any(value.casefold() == "unknown" for value in variety_values) and len(variety_values) > 1:
            flags.append("unknown_and_named_varieties")
        if (
            "variety" in group.columns
            and _has_mixed_missingness(group["variety"])
        ):
            flags.append("missing_variety_label_on_some_rows")
        if len(variety_id_values) > 1:
            flags.append("multiple_variety_ids")
        if (
            "variety_id" in group.columns
            and _has_mixed_missingness(group["variety_id"])
        ):
            flags.append("missing_variety_id_on_some_rows")
        if len(unit_values) > 1:
            flags.append("multiple_price_units")
        if (
            "price_unit" in group.columns
            and _has_mixed_missingness(group["price_unit"])
        ):
            flags.append("missing_price_unit_on_some_rows")
        if len(market_values) > 1:
            flags.append("multiple_market_labels_for_market_id")
        if len(crop_values) > 1:
            flags.append("multiple_crop_labels_for_commodity_id")
        if len(commodity_name_values) > 1:
            flags.append("multiple_commodity_names")

        # Emit only groups where one or more metadata attributes differ.
        if not flags:
            continue

        numeric_values = pd.to_numeric(
            group.get(
                "reported_price_numeric",
                group.get("reported_price", pd.Series(index=group.index, dtype="float64")),
            ),
            errors="coerce",
        ).dropna().unique()
        numeric_values_sorted = sorted(float(value) for value in numeric_values)

        row = dict(zip(group_keys, identity))
        row.update(
            {
                "market_labels": "; ".join(market_values),
                "crop_labels": "; ".join(crop_values),
                "commodity_names": "; ".join(commodity_name_values),
                "row_count": int(len(group)),
                "variety_labels": "; ".join(variety_values),
                "variety_ids": "; ".join(variety_id_values),
                "price_units": "; ".join(unit_values),
                "reported_price_raw_values": _joined_values(
                    group, "reported_price_raw"
                ),
                "price_status_values": _joined_values(
                    group, "price_status"
                ),
                "numeric_price_values": "; ".join(
                    format(value, "g") for value in numeric_values_sorted
                ),
                "distinct_numeric_price_values": int(
                    len(numeric_values_sorted)
                ),
                "raw_response_files": _joined_values(
                    group, "raw_response_file"
                ),
                "metadata_flags": "; ".join(flags),
                "review_status": "review_required",
            }
        )
        rows.append(row)

    return pd.DataFrame(rows, columns=METADATA_REPORT_COLUMNS)


def main(argv: list[str] | None = None) -> int:
    """Run the read-only archive audit and write reports."""
    parser = argparse.ArgumentParser(
        description="Audit the Dharashiv local mandi observation archive."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    df = load_archive(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    coverage = group_quality(df)

    # Include all configured crop/market combinations so zero-reporting
    # combinations remain visible in the coverage report.
    expected_grid = pd.DataFrame(
        [
            {"crop": crop, "market": market, "market_id": market_id}
            for crop in EXPECTED_CROPS
            for market, market_id in EXPECTED_MARKETS.items()
        ]
    )
    if not coverage.empty:
        coverage["market_id"] = coverage["market_id"].astype("string")
    expected_grid["market_id"] = expected_grid["market_id"].astype("string")
    coverage = expected_grid.merge(
        coverage,
        on=["crop", "market", "market_id"],
        how="outer",
    )

    for count_col in [
        "rows",
        "positive_numeric_reported_rows",
        "unique_observation_dates",
        "distinct_varieties",
        "arrivals_rows",
    ]:
        if count_col in coverage.columns:
            coverage[count_col] = coverage[count_col].fillna(0).astype(int)

    duplicates = duplicate_report(df)
    metadata_inconsistencies = metadata_inconsistency_report(df)
    outliers = outlier_report(df)

    coverage.to_csv(args.output_dir / "market_crop_coverage.csv", index=False)
    duplicates.to_csv(args.output_dir / "duplicate_groups.csv", index=False)
    metadata_inconsistencies.to_csv(
        args.output_dir / "metadata_inconsistencies.csv",
        index=False,
    )
    outliers.to_csv(args.output_dir / "price_outlier_flags.csv", index=False)

    valid_price = df["reported_price_numeric"].gt(0)
    summary: dict[str, Any] = {
        "input": str(args.input),
        "rows": int(len(df)),
        "unique_crops": int(df["crop"].nunique(dropna=True)),
        "unique_markets": int(df["market_id"].nunique(dropna=True)),
        "positive_numeric_reported_rows": int(valid_price.sum()),
        "missing_or_invalid_price_rows": int((~valid_price).sum()),
        "invalid_date_rows": int(df["observation_date"].isna().sum()),
        "duplicate_or_conflicting_groups": int(len(duplicates)),
        "metadata_inconsistency_groups": int(len(metadata_inconsistencies)),
        "coverage_metrics_note": (
            "unique_observation_dates includes NR/missing-price rows; "
            "unique_reported_price_dates includes only positive numeric prices."
        ),
        "price_unit_values": _distinct_values(df["price_unit"]),
        "source_values": _distinct_values(df["data_source"]),
        "price_status_counts": {
            str(key): int(value)
            for key, value in df["price_status"].fillna(
                "missing_status"
            ).value_counts().items()
        },
        "arrivals_rows": int(df["arrivals_numeric"].notna().sum()),
        "variety_values": _distinct_values(df["variety"]),
        "outlier_flags_review_only": int(len(outliers)),
        "coverage_note": (
            "Observation-date coverage includes all observations, including NR. "
            "Reported-price coverage includes only positive numeric prices. "
            "Gap fields are elapsed days between consecutive dates; the separate "
            "missing-calendar-days fields count intervening calendar days. "
            "No field infers mandi trading/closure days."
        ),
        "price_semantics_note": (
            "Weekly collector stores the reported value as unverified; do not "
            "rename it modal without endpoint-specific schema confirmation."
        ),
        "arrivals_note": (
            "The weekly commodity-price response used by the collector does "
            "not supply arrivals; missing arrivals are not zero."
        ),
        "metadata_note": (
            "Metadata inconsistencies are review flags only. No variety, "
            "unit, or observation is automatically reconciled or removed."
        ),
    }

    if args.runs.exists():
        runs = pd.read_csv(args.runs, low_memory=False)
        if "status" in runs.columns:
            summary["collection_run_status_counts"] = {
                str(key): int(value)
                for key, value in runs["status"].value_counts(
                    dropna=False
                ).items()
            }
            required_run_columns = {"crop", "market", "status"}
            if required_run_columns.issubset(runs.columns):
                run_summary = (
                    runs.groupby(
                        ["crop", "market", "status"], dropna=False
                    )
                    .size()
                    .reset_index(name="requests")
                )
                run_summary.to_csv(
                    args.output_dir / "collection_run_status.csv",
                    index=False,
                )
    else:
        summary["collection_run_log"] = (
            "not_found; only observation archive can be audited"
        )

    summary_path = args.output_dir / "quality_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n--- DHARASHIV MARKET DATA QUALITY AUDIT ---")
    print(
        f"Rows: {summary['rows']} | Crops: {summary['unique_crops']} "
        f"| Markets: {summary['unique_markets']}"
    )
    print(
        "Positive numeric reported values:",
        summary["positive_numeric_reported_rows"],
    )
    print(
        "Missing/invalid prices:",
        summary["missing_or_invalid_price_rows"],
    )
    print("Invalid dates:", summary["invalid_date_rows"])
    print(
        "Duplicate/conflicting groups:",
        summary["duplicate_or_conflicting_groups"],
    )
    print(
        "Metadata inconsistency groups:",
        summary["metadata_inconsistency_groups"],
    )
    print("Price units:", summary["price_unit_values"])
    print("Arrivals values present:", summary["arrivals_rows"])
    print(
        "IQR outlier flags (review only):",
        summary["outlier_flags_review_only"],
    )

    if not coverage.empty:
        print("\nCoverage by crop and market:")
        print(coverage.to_string(index=False))

    print("\nSaved audit reports to:", args.output_dir)
    print("No source records were removed or modified by this audit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
