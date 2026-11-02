#!/usr/bin/env python3
"""Evaluate simple one-month-ahead baselines on the monthly validation split.

Metrics are calculated in original Rs./quintal units, not standardized units.
This script intentionally evaluates validation only; keep the test split untouched
until the model architecture and training choices are finalized.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if len(actual) == 0:
        raise ValueError("Cannot calculate metrics for an empty group")
    error = predicted - actual
    return {
        "n": int(len(actual)),
        "mae_rs_qtl": float(np.mean(np.abs(error))),
        "rmse_rs_qtl": float(np.sqrt(np.mean(error**2))),
        "mape_pct": float(np.mean(np.abs(error / actual)) * 100.0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--monthly-csv",
        default="data/processed/maharashtra_monthly.csv",
        help="Cleaned Maharashtra monthly source data",
    )
    parser.add_argument(
        "--metadata",
        default="data/training/monthly_maharashtra/metadata_val.csv",
        help="Validation target metadata (test metadata is not used)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/training/monthly_maharashtra/baselines",
        help="Directory for validation predictions and metrics",
    )
    args = parser.parse_args()

    monthly_path = Path(args.monthly_csv)
    metadata_path = Path(args.metadata)
    output_dir = Path(args.output_dir)
    if not monthly_path.exists():
        raise FileNotFoundError(f"Monthly source file not found: {monthly_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Validation metadata not found: {metadata_path}")

    monthly = pd.read_csv(monthly_path)
    metadata = pd.read_csv(metadata_path)
    required_monthly = {"month", "crop", "state", "price_rs_qtl"}
    required_metadata = {"crop", "state", "target_month", "target_price_rs_qtl"}
    if missing := sorted(required_monthly - set(monthly.columns)):
        raise ValueError(f"Monthly CSV is missing columns: {missing}")
    if missing := sorted(required_metadata - set(metadata.columns)):
        raise ValueError(f"Validation metadata is missing columns: {missing}")

    monthly["month"] = pd.to_datetime(monthly["month"], errors="coerce").dt.to_period("M")
    monthly["price_rs_qtl"] = pd.to_numeric(monthly["price_rs_qtl"], errors="coerce")
    if monthly["month"].isna().any():
        raise ValueError("Monthly CSV contains invalid month values")

    usable = monthly.loc[
        monthly["price_rs_qtl"].notna() & (monthly["price_rs_qtl"] > 0),
        ["month", "crop", "state", "price_rs_qtl"],
    ].copy()
    if usable.duplicated(["crop", "state", "month"]).any():
        raise ValueError("Duplicate crop/state/month prices found in monthly CSV")

    price_lookup = {
        (str(row.crop), str(row.state), row.month): float(row.price_rs_qtl)
        for row in usable.itertuples(index=False)
    }

    metadata["target_period"] = pd.PeriodIndex(metadata["target_month"], freq="M")
    metadata["target_price_rs_qtl"] = pd.to_numeric(
        metadata["target_price_rs_qtl"], errors="coerce"
    )
    if metadata["target_price_rs_qtl"].isna().any() or (
        metadata["target_price_rs_qtl"] <= 0
    ).any():
        raise ValueError("Validation metadata contains missing or nonpositive target prices")

    prediction_rows: list[dict[str, object]] = []
    baseline_offsets = {
        "last_month_naive": 1,
        "seasonal_naive_12m": 12,
    }
    for row in metadata.itertuples(index=False):
        crop, state = str(row.crop), str(row.state)
        target_period = row.target_period
        actual = float(row.target_price_rs_qtl)
        for baseline_name, offset in baseline_offsets.items():
            source_period = target_period - offset
            source_key = (crop, state, source_period)
            if source_key not in price_lookup:
                raise ValueError(
                    f"Missing historical price for {crop}/{state} at {source_period}; "
                    "cannot fairly evaluate all validation targets."
                )
            prediction_rows.append({
                "crop": crop,
                "state": state,
                "target_month": str(target_period),
                "baseline": baseline_name,
                "source_month": str(source_period),
                "actual_price_rs_qtl": actual,
                "predicted_price_rs_qtl": price_lookup[source_key],
            })

    predictions = pd.DataFrame(prediction_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "validation_predictions.csv", index=False)

    by_crop_rows: list[dict[str, object]] = []
    for (baseline, crop), group in predictions.groupby(["baseline", "crop"], sort=True):
        row = {"baseline": baseline, "crop": crop}
        row.update(metrics(
            group["actual_price_rs_qtl"].to_numpy(),
            group["predicted_price_rs_qtl"].to_numpy(),
        ))
        by_crop_rows.append(row)
    by_crop = pd.DataFrame(by_crop_rows)
    by_crop.to_csv(output_dir / "validation_metrics_by_crop.csv", index=False)

    overall: dict[str, dict[str, float | int]] = {}
    for baseline, group in predictions.groupby("baseline", sort=True):
        overall[str(baseline)] = metrics(
            group["actual_price_rs_qtl"].to_numpy(),
            group["predicted_price_rs_qtl"].to_numpy(),
        )
    import json
    with (output_dir / "validation_metrics_overall.json").open("w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2)

    print("\n--- VALIDATION BASELINES (Rs./quintal) ---")
    print(by_crop.to_string(index=False, float_format=lambda value: f"{value:,.2f}"))
    print("\n--- OVERALL VALIDATION METRICS ---")
    for baseline, result in overall.items():
        print(
            f"{baseline}: n={result['n']}, "
            f"MAE={result['mae_rs_qtl']:.2f}, "
            f"RMSE={result['rmse_rs_qtl']:.2f}, "
            f"MAPE={result['mape_pct']:.2f}%"
        )
    print(f"\nSaved validation baseline outputs to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())