#!/usr/bin/env python3
"""Clean Agmarknet/UPAG India mandi extracts for CropBalance AI.

This script intentionally preserves missing monthly values. It does not
interpolate prices or convert missing observations to zero.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

CORE_CROPS = {
    "wheat": {"wheat"},
    "rice": {"rice"},
    "tomato": {"tomato"},
    "onion": {"onion"},
    "potato": {"potato"},
    "soybean": {"soybean", "soyabean"},
    "maize": {"maize"},
}

MONTHLY_REQUIRED = {
    "month", "commodity_id", "commodity", "group", "state",
    "arrivals_t", "price_rs_qtl",
}
MARKET_REQUIRED = {
    "date", "commodity", "state", "district", "market",
    "mkt_id", "arrival_t", "modal_price_rs_qtl",
}


def normalize_name(value: object) -> str:
    """Normalize commodity spelling for exact alias matching."""
    return re.sub(r"[^a-z0-9]", "", str(value).strip().lower())


ALIAS_TO_CROP = {
    normalize_name(alias): crop
    for crop, aliases in CORE_CROPS.items()
    for alias in aliases
}


def require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def parse_states(states_arg: str) -> list[str] | None:
    """Return None for all states, otherwise a normalized requested list."""
    if states_arg.strip().lower() in {"all", "*"}:
        return None
    values = [value.strip() for value in states_arg.split(",") if value.strip()]
    if not values:
        raise ValueError("--states must be 'all' or a comma-separated state list")
    return values


def clean_monthly(input_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not input_path.exists():
        raise FileNotFoundError(f"Monthly source file not found: {input_path}")

    raw = pd.read_csv(input_path)
    require_columns(raw, MONTHLY_REQUIRED, "Monthly source")

    raw["month"] = pd.to_datetime(raw["month"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    bad_months = int(raw["month"].isna().sum())
    if bad_months:
        raise ValueError(f"Monthly source contains {bad_months} invalid month values")

    raw["crop"] = raw["commodity"].map(lambda x: ALIAS_TO_CROP.get(normalize_name(x)))
    clean = raw.loc[raw["crop"].notna()].copy()

    clean["commodity_id"] = pd.to_numeric(clean["commodity_id"], errors="coerce")
    clean["arrivals_t"] = pd.to_numeric(clean["arrivals_t"], errors="coerce")
    clean["price_rs_qtl"] = pd.to_numeric(clean["price_rs_qtl"], errors="coerce")

    # Invalid values become missing; they are never turned into zero.
    clean.loc[clean["arrivals_t"] < 0, "arrivals_t"] = np.nan
    clean.loc[clean["price_rs_qtl"] <= 0, "price_rs_qtl"] = np.nan

    key = ["month", "commodity_id", "state"]
    duplicate_count = int(clean.duplicated(key, keep=False).sum())
    if duplicate_count:
        examples = clean.loc[clean.duplicated(key, keep=False), key].head(10)
        raise ValueError(
            "Monthly source has duplicate month/commodity/state keys. "
            f"Resolve these before training. Examples:\n{examples.to_string(index=False)}"
        )

    clean["price_missing"] = clean["price_rs_qtl"].isna().astype("int8")
    clean["arrivals_missing"] = clean["arrivals_t"].isna().astype("int8")

    # Keep original commodity spelling for traceability; retain only useful columns.
    clean = clean[[
        "month", "commodity_id", "crop", "commodity", "group", "state",
        "arrivals_t", "arrivals_missing", "price_rs_qtl", "price_missing",
    ]].sort_values(["crop", "state", "month"]).reset_index(drop=True)

    coverage = build_monthly_coverage(clean)
    return clean, coverage


def build_monthly_coverage(clean: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (crop, state), group in clean.groupby(["crop", "state"], sort=True):
        group = group.sort_values("month")
        start = group["month"].min()
        end = group["month"].max()
        expected = len(pd.date_range(start, end, freq="MS"))
        observed_months = group["month"].nunique()
        valid_prices = int(group["price_rs_qtl"].notna().sum())
        valid_arrivals = int(group["arrivals_t"].notna().sum())
        rows.append({
            "crop": crop,
            "state": state,
            "first_month": start.strftime("%Y-%m"),
            "last_month": end.strftime("%Y-%m"),
            "expected_months": expected,
            "rows_present": int(observed_months),
            "missing_calendar_months": int(expected - observed_months),
            "valid_price_months": valid_prices,
            "missing_price_months": int(expected - valid_prices),
            "price_coverage_pct": round(100 * valid_prices / expected, 2) if expected else 0.0,
            "valid_arrival_months": valid_arrivals,
            "missing_arrival_months": int(expected - valid_arrivals),
            "arrival_coverage_pct": round(100 * valid_arrivals / expected, 2) if expected else 0.0,
        })
    return pd.DataFrame(rows)


def clean_market_daily(input_path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    if not input_path.exists():
        raise FileNotFoundError(f"Market-level source file not found: {input_path}")

    raw = pd.read_csv(input_path)
    require_columns(raw, MARKET_REQUIRED, "Market daily source")

    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    bad_dates = int(raw["date"].isna().sum())
    if bad_dates:
        raise ValueError(f"Market daily source contains {bad_dates} invalid dates")

    raw["crop"] = raw["commodity"].map(lambda x: ALIAS_TO_CROP.get(normalize_name(x)))
    frame = raw.loc[raw["crop"].notna()].copy()

    frame["arrival_t"] = pd.to_numeric(frame["arrival_t"], errors="coerce")
    frame["modal_price_rs_qtl"] = pd.to_numeric(frame["modal_price_rs_qtl"], errors="coerce")
    frame.loc[frame["arrival_t"] < 0, "arrival_t"] = np.nan
    frame.loc[frame["modal_price_rs_qtl"] <= 0, "modal_price_rs_qtl"] = np.nan

    # Market observations without a valid target price are excluded from this
    # modelling table and counted in the report. Missing arrivals are retained.
    before = len(frame)
    frame = frame.loc[frame["modal_price_rs_qtl"].notna()].copy()
    excluded_no_price = before - len(frame)

    key = ["date", "mkt_id", "crop", "commodity"]
    dupes = int(frame.duplicated(key, keep=False).sum())
    if dupes:
        sample = frame.loc[frame.duplicated(key, keep=False), key].head(10)
        raise ValueError(
            "Market sample has duplicate date/market/crop/commodity keys. "
            f"Examples:\n{sample.to_string(index=False)}"
        )

    frame = frame[[
        "date", "crop", "commodity", "state", "district", "market", "mkt_id",
        "arrival_t", "modal_price_rs_qtl",
    ]].sort_values(["crop", "state", "market", "date"]).reset_index(drop=True)

    report = {
        "rows_after_crop_filter": int(before),
        "rows_with_valid_modal_price": int(len(frame)),
        "rows_excluded_for_missing_or_invalid_price": int(excluded_no_price),
        "unique_dates": int(frame["date"].nunique()),
        "unique_markets": int(frame["mkt_id"].nunique()),
        "unique_states": int(frame["state"].nunique()),
        "sampling_note": "Source market file is a sampled series; do not treat it as complete daily data.",
    }
    return frame, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--monthly-input",
        default="data/external/india-mandi-arrivals/data/agmarknet/state_monthly.csv.gz",
        help="Path to Agmarknet state_monthly CSV/CSV.GZ",
    )
    parser.add_argument(
        "--market-input",
        default="data/external/india-mandi-arrivals/data/agmarknet/market_daily_sample.csv.gz",
        help="Path to market_daily_sample CSV/CSV.GZ",
    )
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument(
        "--states", default="all",
        help="State filter for india_monthly_clean.csv (default all). Use Maharashtra for a single state.",
    )
    parser.add_argument("--skip-market", action="store_true", help="Skip market-level sample processing")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    states = parse_states(args.states)
    monthly, coverage = clean_monthly(Path(args.monthly_input))
    if states is not None:
        wanted = {s.casefold() for s in states}
        monthly_output = monthly.loc[monthly["state"].str.casefold().isin(wanted)].copy()
        if monthly_output.empty:
            raise ValueError(f"No monthly rows found for requested states: {states}")
    else:
        monthly_output = monthly

    monthly_output.to_csv(output_dir / "india_monthly_clean.csv", index=False)
    maharashtra = monthly.loc[monthly["state"].str.casefold() == "maharashtra"].copy()
    maharashtra.to_csv(output_dir / "maharashtra_monthly.csv", index=False)
    coverage.to_csv(output_dir / "monthly_coverage.csv", index=False)

    manifest: dict[str, object] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "crop_aliases": {k: sorted(v) for k, v in CORE_CROPS.items()},
        "monthly_source": str(Path(args.monthly_input)),
        "monthly_rows_core_crops_all_states": int(len(monthly)),
        "monthly_rows_written": int(len(monthly_output)),
        "maharashtra_monthly_rows": int(len(maharashtra)),
        "monthly_crop_counts_all_states": {
            str(k): int(v) for k, v in monthly.groupby("crop").size().to_dict().items()
        },
        "price_missing_values_preserved": True,
        "no_price_interpolation_performed": True,
    }

    if not args.skip_market:
        market_path = Path(args.market_input)
        if market_path.exists():
            market, market_report = clean_market_daily(market_path)
            market.to_csv(output_dir / "market_sample_clean.csv", index=False)
            manifest["market_source"] = str(market_path)
            manifest["market_report"] = market_report
        else:
            print(
                f"WARNING: Market sample not found at {market_path}; "
                "skipping it. Use --skip-market to silence this on purpose.",
                file=sys.stderr,
            )

    with open(output_dir / "preprocessing_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("\n--- CLEANING SUMMARY ---")
    print("All-state core-crop rows:", len(monthly))
    print("Rows written to india_monthly_clean.csv:", len(monthly_output))
    print("Maharashtra rows:", len(maharashtra))
    print("Crop counts:")
    print(monthly.groupby("crop").size().to_string())
    print("\nSaved outputs to:", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
