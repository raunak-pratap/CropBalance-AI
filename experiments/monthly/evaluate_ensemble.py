#!/usr/bin/env python3
"""Evaluate multi-seed LSTM ensembles on the monthly validation split only.

Reads the saved validation_predictions.csv files from five trained seeds, checks
that all files predict the exact same crop/state/month targets and actual prices,
then compares each seed, row-wise mean and row-wise median against the saved
last-month-naive baseline. It never loads test data or test metadata.

Default directories assume this script is run from the CropBalance-AI project root.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

KEYS = ["crop", "state", "target_month"]
PREDICTION_COLUMNS = ["actual_price_rs_qtl", "predicted_price_rs_qtl"]
DEFAULT_RUNS = {
    "seed_42": "data/training/monthly_maharashtra/monthly_lstm",
    "seed_7": "data/training/monthly_maharashtra/monthly_lstm_seed_7",
    "seed_21": "data/training/monthly_maharashtra/monthly_lstm_seed_21",
    "seed_123": "data/training/monthly_maharashtra/monthly_lstm_seed_123",
    "seed_2026": "data/training/monthly_maharashtra/monthly_lstm_seed_2026",
}


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.ndim != 1 or predicted.ndim != 1 or len(actual) == 0 or len(actual) != len(predicted):
        raise ValueError("Metric inputs must be non-empty, one-dimensional, and equal length")
    if not (np.isfinite(actual).all() and np.isfinite(predicted).all()):
        raise ValueError("Metric inputs contain NaN or infinite values")
    if np.any(actual <= 0):
        raise ValueError("Actual prices must be positive to calculate MAPE")
    error = predicted - actual
    return {
        "n": int(len(actual)),
        "mae_rs_qtl": float(np.mean(np.abs(error))),
        "rmse_rs_qtl": float(np.sqrt(np.mean(error**2))),
        "mape_pct": float(np.mean(np.abs(error / actual)) * 100.0),
    }


def read_model_predictions(label: str, directory: Path) -> pd.DataFrame:
    path = directory / "validation_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing validation predictions for {label}: {path}")
    frame = pd.read_csv(path)
    required = set(KEYS + PREDICTION_COLUMNS)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    frame = frame[KEYS + PREDICTION_COLUMNS].copy()
    for col in KEYS:
        frame[col] = frame[col].astype(str)
    for col in PREDICTION_COLUMNS:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if frame[PREDICTION_COLUMNS].isna().any().any():
        raise ValueError(f"{path} contains missing or non-numeric prices")
    if not np.isfinite(frame[PREDICTION_COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError(f"{path} contains infinite prices")
    if (frame["actual_price_rs_qtl"] <= 0).any():
        raise ValueError(f"{path} contains nonpositive actual prices")
    if frame.duplicated(KEYS).any():
        duplicated = frame.loc[frame.duplicated(KEYS, keep=False), KEYS].head(5)
        raise ValueError(f"{path} has duplicate validation targets:\n{duplicated}")
    return frame.sort_values(KEYS).reset_index(drop=True).rename(
        columns={
            "actual_price_rs_qtl": f"actual_{label}",
            "predicted_price_rs_qtl": f"pred_{label}",
        }
    )


def read_baseline(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Naive baseline predictions not found: {path}")
    frame = pd.read_csv(path)
    required = set(KEYS + ["baseline", "actual_price_rs_qtl", "predicted_price_rs_qtl"])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    frame = frame.loc[frame["baseline"].astype(str) == "last_month_naive", KEYS + PREDICTION_COLUMNS].copy()
    if frame.empty:
        raise ValueError(f"No last_month_naive rows found in {path}")
    for col in KEYS:
        frame[col] = frame[col].astype(str)
    for col in PREDICTION_COLUMNS:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if frame[PREDICTION_COLUMNS].isna().any().any() or not np.isfinite(frame[PREDICTION_COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError(f"{path} contains missing or non-finite baseline prices")
    if frame.duplicated(KEYS).any():
        raise ValueError(f"{path} has duplicate last_month_naive validation targets")
    return frame.sort_values(KEYS).reset_index(drop=True).rename(
        columns={
            "actual_price_rs_qtl": "actual_baseline",
            "predicted_price_rs_qtl": "pred_last_month_naive",
        }
    )


def add_metrics_rows(
    predictions: pd.DataFrame,
    predictor_columns: Iterable[tuple[str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall_rows: list[dict[str, object]] = []
    crop_rows: list[dict[str, object]] = []
    for predictor_name, prediction_column in predictor_columns:
        actual_column = "actual_baseline" if predictor_name == "last_month_naive" else "actual_seed_42"
        # All actual columns have already been checked for equality, so one aligned
        # reference column is sufficient for all model and ensemble metrics.
        actual = predictions[actual_column].to_numpy(dtype=float)
        result = {"predictor": predictor_name}
        result.update(metrics(actual, predictions[prediction_column].to_numpy(dtype=float)))
        overall_rows.append(result)
        for crop, group in predictions.groupby("crop", sort=True):
            crop_actual = group[actual_column].to_numpy(dtype=float)
            crop_result = {"predictor": predictor_name, "crop": str(crop)}
            crop_result.update(metrics(crop_actual, group[prediction_column].to_numpy(dtype=float)))
            crop_rows.append(crop_result)
    return pd.DataFrame(overall_rows), pd.DataFrame(crop_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", default="data/training/monthly_maharashtra")
    parser.add_argument("--baseline-predictions", default="data/training/monthly_maharashtra/baselines/validation_predictions.csv")
    parser.add_argument("--output-dir", default="data/training/monthly_maharashtra/ensemble_evaluation")
    parser.add_argument(
        "--run",
        action="append",
        default=None,
        metavar="LABEL=RELATIVE_DIRECTORY",
        help="Override/add a model run. May be repeated. Defaults to the five known seed directories.",
    )
    args = parser.parse_args()

    training_dir = Path(args.training_dir)
    runs = dict(DEFAULT_RUNS)
    if args.run:
        runs = {}
        for item in args.run:
            if "=" not in item:
                raise ValueError(f"Invalid --run {item!r}; expected LABEL=RELATIVE_DIRECTORY")
            label, directory = item.split("=", 1)
            label, directory = label.strip(), directory.strip()
            if not label or not directory or label in runs:
                raise ValueError(f"Invalid or duplicate --run value: {item!r}")
            runs[label] = directory
    if len(runs) < 2:
        raise ValueError("At least two model runs are required for ensemble evaluation")

    merged: pd.DataFrame | None = None
    reference_label: str | None = None
    actual_columns: list[str] = []
    prediction_columns: list[tuple[str, str]] = []

    for label, relative_directory in runs.items():
        directory = Path(relative_directory)
        if not directory.is_absolute():
            directory = Path.cwd() / directory
        model_frame = read_model_predictions(label, directory)
        actual_col, pred_col = f"actual_{label}", f"pred_{label}"
        actual_columns.append(actual_col)
        prediction_columns.append((label, pred_col))
        if merged is None:
            merged = model_frame
            reference_label = label
            continue
        assert reference_label is not None
        if len(model_frame) != len(merged) or not model_frame[KEYS].equals(merged[KEYS]):
            raise ValueError(
                f"Validation target keys/order differ between {reference_label} and {label}. "
                "All runs must use the same validation metadata."
            )
        if not np.allclose(
            model_frame[actual_col].to_numpy(dtype=float),
            merged[f"actual_{reference_label}"].to_numpy(dtype=float),
            rtol=1e-7,
            atol=1e-4,
        ):
            raise ValueError(f"Actual validation prices differ between {reference_label} and {label}")
        merged[pred_col] = model_frame[pred_col].to_numpy(dtype=float)

    assert merged is not None and reference_label is not None
    baseline = read_baseline(Path(args.baseline_predictions))
    if len(baseline) != len(merged) or not baseline[KEYS].equals(merged[KEYS]):
        raise ValueError("Last-month baseline targets do not exactly match the LSTM validation targets")
    if not np.allclose(
        baseline["actual_baseline"].to_numpy(dtype=float),
        merged[f"actual_{reference_label}"].to_numpy(dtype=float),
        rtol=1e-7,
        atol=1e-4,
    ):
        raise ValueError("Actual validation prices in baseline and LSTM files do not match")
    merged = merged.merge(baseline, on=KEYS, how="inner", validate="one_to_one")

    # Build row-wise prediction ensembles; these are averages/medians of predictions,
    # not averages of model metrics.
    seed_pred_columns = [column for _, column in prediction_columns]
    merged["pred_mean_ensemble"] = merged[seed_pred_columns].mean(axis=1)
    merged["pred_median_ensemble"] = merged[seed_pred_columns].median(axis=1)

    # Use the baseline's actual column as the common actual reference after checking
    # that every model file contains the same target prices.
    metric_predictors = [("last_month_naive", "pred_last_month_naive")] + prediction_columns + [
        ("mean_ensemble", "pred_mean_ensemble"),
        ("median_ensemble", "pred_median_ensemble"),
    ]
    # make every row use baseline actual in the metric function
    merged["actual_seed_42"] = merged["actual_baseline"]
    overall, by_crop = add_metrics_rows(merged, metric_predictors)

    # Summarize variability across the individual model runs, distinct from the
    # performance of the row-wise ensembles.
    individual = overall.loc[overall["predictor"].isin([label for label, _ in prediction_columns])]
    seed_stats = {
        metric: {
            "mean": float(individual[metric].mean()),
            "std_sample": float(individual[metric].std(ddof=1)),
            "min": float(individual[metric].min()),
            "max": float(individual[metric].max()),
        }
        for metric in ("mae_rs_qtl", "rmse_rs_qtl", "mape_pct")
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    display_columns = KEYS + ["actual_baseline", "pred_last_month_naive"] + seed_pred_columns + [
        "pred_mean_ensemble", "pred_median_ensemble"
    ]
    merged[display_columns].to_csv(output_dir / "validation_ensemble_predictions.csv", index=False)
    overall.to_csv(output_dir / "validation_metrics_overall.csv", index=False)
    by_crop.to_csv(output_dir / "validation_metrics_by_crop.csv", index=False)

    summary = {
        "evaluation_split": "validation_only",
        "test_data_loaded": False,
        "n_validation_targets": int(len(merged)),
        "model_runs": list(runs.keys()),
        "ensemble_methods": ["row_wise_mean", "row_wise_median"],
        "individual_seed_metric_summary": seed_stats,
        "overall_metrics": overall.to_dict(orient="records"),
        "notes": [
            "All model prediction files were checked for identical crop/state/month keys and actual prices.",
            "Last-month baseline targets were checked against the same validation targets.",
            "Ensemble predictions are row-wise mean and median across individual model predictions.",
            "Test split was not loaded or evaluated.",
        ],
    }
    with (output_dir / "experiment_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n--- VALIDATION OVERALL METRICS (Rs./quintal) ---")
    print(overall.to_string(index=False, float_format=lambda value: f"{value:,.2f}"))
    print("\n--- VALIDATION MAE BY CROP (Rs./quintal) ---")
    print(by_crop.pivot(index="crop", columns="predictor", values="mae_rs_qtl").to_string(float_format=lambda value: f"{value:,.2f}"))
    print("\n--- VARIABILITY ACROSS INDIVIDUAL SEEDS ---")
    for metric, stats in seed_stats.items():
        unit = "%" if metric == "mape_pct" else " Rs./quintal"
        print(f"{metric}: mean={stats['mean']:.2f}{unit}, sample_std={stats['std_sample']:.2f}{unit}, min={stats['min']:.2f}, max={stats['max']:.2f}")
    print(f"\nSaved validation ensemble evaluation to: {output_dir}")
    print("Test split remains untouched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
