#!/usr/bin/env python3
"""One-time final evaluation of the frozen crop-aware monthly hybrid.

Frozen routing rule (selected using validation data and not changed here):
  * onion, tomato: median prediction from the five saved LSTM seeds
  * all other crops: last-month-naive prediction

This script loads the held-out test arrays and evaluates the rule exactly once.
It does not train, tune, or select models using test results. Do not change the
routing rule or hyperparameters in response to these test metrics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

KEYS = ["crop", "state", "target_month"]
LSTM_CROPS = {"onion", "tomato"}
DEFAULT_RUNS = {
    "seed_42": "monthly_lstm",
    "seed_7": "monthly_lstm_seed_7",
    "seed_21": "monthly_lstm_seed_21",
    "seed_123": "monthly_lstm_seed_123",
    "seed_2026": "monthly_lstm_seed_2026",
}


class MonthlyLSTM(nn.Module):
    """Architecture matching train_monthly_lstm.py."""

    def __init__(self, input_size: int = 13, hidden_size: int = 32, dropout: float = 0.10):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence_output, _ = self.lstm(x)
        return self.head(sequence_output[:, -1, :])


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def inverse_scale(
    values: np.ndarray,
    metadata: pd.DataFrame,
    scalers: dict[str, dict[str, Any]],
) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    if len(values) != len(metadata):
        raise ValueError("Cannot inverse-scale: prediction count and metadata rows differ")
    output = np.empty(len(metadata), dtype=float)
    for i, row in enumerate(metadata.itertuples(index=False)):
        key = str(row.series_key)
        if key not in scalers:
            raise KeyError(f"Missing scaler for {key!r}")
        scaler = scalers[key]
        output[i] = values[i] * float(scaler["price_std_train"]) + float(scaler["price_mean_train"])
    return output


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.ndim != 1 or predicted.ndim != 1 or len(actual) == 0 or len(actual) != len(predicted):
        raise ValueError("Metric inputs must be non-empty, 1-D, and equal length")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Metric inputs contain NaN or infinite values")
    if np.any(actual <= 0):
        raise ValueError("Actual prices must be positive for MAPE")
    error = predicted - actual
    return {
        "n": int(len(actual)),
        "mae_rs_qtl": float(np.mean(np.abs(error))),
        "rmse_rs_qtl": float(np.sqrt(np.mean(error ** 2))),
        "mape_pct": float(np.mean(np.abs(error / actual)) * 100.0),
    }


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing trained checkpoint: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # compatibility with older PyTorch versions
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Unexpected checkpoint format: {path}")
    return checkpoint


def predict_with_checkpoint(
    checkpoint_path: Path,
    x_test: np.ndarray,
    metadata: pd.DataFrame,
    scalers: dict[str, dict[str, Any]],
    device: torch.device,
) -> np.ndarray:
    checkpoint = load_checkpoint(checkpoint_path)
    config = checkpoint.get("model_config", {})
    input_size = int(config.get("input_size", x_test.shape[2]))
    hidden_size = int(config.get("hidden_size", 32))
    dropout = float(config.get("dropout", 0.10))
    if input_size != x_test.shape[2]:
        raise ValueError(
            f"Checkpoint {checkpoint_path} expects {input_size} features, "
            f"but X_test has {x_test.shape[2]}"
        )
    model = MonthlyLSTM(input_size=input_size, hidden_size=hidden_size, dropout=dropout)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    x_tensor = torch.from_numpy(x_test.astype(np.float32, copy=False)).to(device)
    with torch.no_grad():
        pred_scaled = model(x_tensor).detach().cpu().numpy().reshape(-1)
    return inverse_scale(pred_scaled, metadata, scalers)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", default="data/training/monthly_maharashtra")
    parser.add_argument(
        "--model-root", default="data/training/monthly_maharashtra",
        help="Directory containing monthly_lstm and monthly_lstm_seed_* folders",
    )
    parser.add_argument(
        "--output-dir", default="data/training/monthly_maharashtra/frozen_hybrid_test_evaluation"
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or cuda")
    args = parser.parse_args()

    training_dir = Path(args.training_dir)
    model_root = Path(args.model_root)
    output_dir = Path(args.output_dir)
    needed = [
        training_dir / "X_test.npy",
        training_dir / "y_test.npy",
        training_dir / "metadata_test.csv",
        training_dir / "scalers.json",
    ]
    missing = [str(path) for path in needed if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing test artifacts:\n- " + "\n- ".join(missing))

    # This is the one-time held-out evaluation: test arrays are intentionally loaded here.
    x_test = np.load(training_dir / "X_test.npy")
    y_test = np.load(training_dir / "y_test.npy")
    metadata = pd.read_csv(training_dir / "metadata_test.csv")
    with (training_dir / "scalers.json").open(encoding="utf-8") as f:
        scalers: dict[str, dict[str, Any]] = json.load(f)

    required_meta = set(KEYS + ["series_key", "target_price_rs_qtl"])
    absent = sorted(required_meta - set(metadata.columns))
    if absent:
        raise ValueError(f"metadata_test.csv is missing columns: {absent}")
    if x_test.ndim != 3 or x_test.shape[1:] != (12, 13):
        raise ValueError(f"Expected X_test shape (samples, 12, 13); received {x_test.shape}")
    if len(x_test) != len(metadata) or y_test.shape != (len(metadata), 1):
        raise ValueError("X_test, y_test, and metadata_test row counts/shapes do not align")
    if not np.isfinite(x_test).all() or not np.isfinite(y_test).all():
        raise ValueError("Test arrays contain NaN or infinite values")
    for column in KEYS + ["series_key"]:
        metadata[column] = metadata[column].astype(str).str.strip()
    metadata["target_price_rs_qtl"] = pd.to_numeric(metadata["target_price_rs_qtl"], errors="coerce")
    if metadata["target_price_rs_qtl"].isna().any() or not np.isfinite(metadata["target_price_rs_qtl"]).all():
        raise ValueError("Test metadata contains missing or non-finite actual prices")
    if (metadata["target_price_rs_qtl"] <= 0).any():
        raise ValueError("Test actual prices must be positive")
    if metadata.duplicated(KEYS).any():
        raise ValueError("Test metadata contains duplicate crop/state/month targets")
    if not metadata["crop"].str.lower().isin(
        {"maize", "onion", "potato", "rice", "soybean", "tomato", "wheat"}
    ).all():
        raise ValueError("Unexpected crop found in test metadata")

    actual = metadata["target_price_rs_qtl"].to_numpy(dtype=float)
    y_actual = inverse_scale(y_test.reshape(-1), metadata, scalers)
    if not np.allclose(y_actual, actual, rtol=1e-5, atol=1e-3):
        raise ValueError(
            "y_test does not inverse-transform to metadata actual prices. "
            "Check that arrays, metadata, and scalers came from the same preprocessing run."
        )

    # Last feature 0 is the train-scaled price at the final input month (t-1).
    # Inverting it yields the last-month-naive forecast for target month t.
    last_month_scaled = x_test[:, -1, 0]
    naive = inverse_scale(last_month_scaled, metadata, scalers)

    run_dirs = {label: model_root / directory for label, directory in DEFAULT_RUNS.items()}
    device = choose_device(args.device)
    print(f"Device: {device}")
    print(f"Held-out test samples: {len(metadata)}")
    print(f"Test target months: {metadata['target_month'].min()} to {metadata['target_month'].max()}")
    print("Frozen routing: onion/tomato = median of five LSTMs; all other crops = last-month naive")
    print("No training or tuning is performed by this script.")

    prediction_frame = metadata[KEYS].copy()
    prediction_frame["actual_price_rs_qtl"] = actual
    prediction_frame["pred_last_month_naive"] = naive

    seed_prediction_columns: list[str] = []
    for label, run_dir in run_dirs.items():
        checkpoint_path = run_dir / "best_model.pt"
        pred = predict_with_checkpoint(checkpoint_path, x_test, metadata, scalers, device)
        column = f"pred_{label}"
        prediction_frame[column] = pred
        seed_prediction_columns.append(column)

    prediction_frame["pred_mean_ensemble"] = prediction_frame[seed_prediction_columns].mean(axis=1)
    prediction_frame["pred_median_ensemble"] = prediction_frame[seed_prediction_columns].median(axis=1)
    use_lstm = prediction_frame["crop"].str.lower().isin(LSTM_CROPS)
    prediction_frame["hybrid_method"] = np.where(
        use_lstm, "median_lstm_ensemble", "last_month_naive"
    )
    prediction_frame["pred_frozen_hybrid"] = np.where(
        use_lstm,
        prediction_frame["pred_median_ensemble"],
        prediction_frame["pred_last_month_naive"],
    )

    predictors = [
        ("last_month_naive", "pred_last_month_naive"),
        ("median_ensemble", "pred_median_ensemble"),
        ("frozen_crop_aware_hybrid", "pred_frozen_hybrid"),
    ]
    overall_rows: list[dict[str, Any]] = []
    crop_rows: list[dict[str, Any]] = []
    for name, column in predictors:
        row: dict[str, Any] = {"predictor": name}
        row.update(metrics(actual, prediction_frame[column].to_numpy(dtype=float)))
        overall_rows.append(row)
        for crop, group in prediction_frame.groupby("crop", sort=True):
            crop_row: dict[str, Any] = {"predictor": name, "crop": crop}
            crop_row.update(
                metrics(
                    group["actual_price_rs_qtl"].to_numpy(dtype=float),
                    group[column].to_numpy(dtype=float),
                )
            )
            crop_rows.append(crop_row)

    overall = pd.DataFrame(overall_rows)
    by_crop = pd.DataFrame(crop_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_frame.to_csv(output_dir / "frozen_hybrid_test_predictions.csv", index=False)
    overall.to_csv(output_dir / "test_metrics_overall.csv", index=False)
    by_crop.to_csv(output_dir / "test_metrics_by_crop.csv", index=False)

    hybrid_row = overall.loc[overall["predictor"] == "frozen_crop_aware_hybrid"].iloc[0]
    naive_row = overall.loc[overall["predictor"] == "last_month_naive"].iloc[0]
    improvement = (
        (float(naive_row["mae_rs_qtl"]) - float(hybrid_row["mae_rs_qtl"]))
        / float(naive_row["mae_rs_qtl"]) * 100.0
    )
    summary = {
        "evaluation_split": "held_out_test",
        "n_test_targets": int(len(prediction_frame)),
        "test_month_start": str(prediction_frame["target_month"].min()),
        "test_month_end": str(prediction_frame["target_month"].max()),
        "model_runs": list(DEFAULT_RUNS.keys()),
        "frozen_routing_rule": {
            "onion": "median_ensemble_of_five_saved_lstm_seeds",
            "tomato": "median_ensemble_of_five_saved_lstm_seeds",
            "all_other_crops": "last_month_naive",
        },
        "overall_metrics": overall.to_dict(orient="records"),
        "hybrid_mae_improvement_vs_naive_pct": improvement,
        "test_used_for_training_or_model_selection": False,
        "warning": (
            "This is the final held-out evaluation of the frozen validation-selected rule. "
            "Do not tune, change crop routing, select a seed, or adjust hyperparameters based on these test results."
        ),
    }
    with (output_dir / "test_evaluation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n--- HELD-OUT TEST METRICS (Rs./quintal) ---")
    print(overall.to_string(index=False, formatters={
        "mae_rs_qtl": "{:,.2f}".format,
        "rmse_rs_qtl": "{:,.2f}".format,
        "mape_pct": "{:.2f}".format,
    }))
    print("\n--- HELD-OUT TEST MAE BY CROP (Rs./quintal) ---")
    print(by_crop.pivot(index="crop", columns="predictor", values="mae_rs_qtl").to_string(float_format=lambda v: f"{v:,.2f}"))
    print(f"\nFrozen hybrid MAE improvement vs naive: {improvement:.2f}%")
    print(f"Saved final test evaluation to: {output_dir}")
    print("IMPORTANT: Do not tune or alter the frozen strategy using these test results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
