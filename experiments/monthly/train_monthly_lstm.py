#!/usr/bin/env python3
"""Train a small shared LSTM for the CropBalance Maharashtra monthly experiment.

Important experiment rules:
- Uses only train and validation arrays; it never reads the test arrays/metadata.
- Selects the checkpoint using validation MAE in original Rs./quintal units.
- Compares against the previously generated last-month-naive baseline.
- Keeps the existing daily forecasting pipeline untouched.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class MonthlyLSTM(nn.Module):
    """Compact shared model; crop identity is already encoded in the inputs."""

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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_arrays(training_dir: Path) -> tuple[np.ndarray, ...]:
    # Deliberately do not load X_test.npy, y_test.npy, or metadata_test.csv.
    names = ("X_train", "y_train", "X_val", "y_val")
    arrays = tuple(np.load(training_dir / f"{name}.npy") for name in names)
    x_train, y_train, x_val, y_val = arrays
    for name, array in zip(names, arrays):
        if not np.isfinite(array).all():
            raise ValueError(f"{name}.npy contains NaN or infinite values")
    if x_train.ndim != 3 or x_val.ndim != 3:
        raise ValueError("X arrays must have shape (samples, lookback, features)")
    if y_train.shape != (len(x_train), 1) or y_val.shape != (len(x_val), 1):
        raise ValueError("y arrays must have shape (samples, 1) and align with X arrays")
    if x_train.shape[1:] != x_val.shape[1:]:
        raise ValueError("Train and validation input shapes do not match")
    return arrays


def inverse_scale(
    scaled_values: np.ndarray,
    metadata: pd.DataFrame,
    scalers: dict[str, dict[str, Any]],
) -> np.ndarray:
    if len(scaled_values) != len(metadata):
        raise ValueError("Prediction count does not match metadata row count")
    result = np.empty(len(metadata), dtype=float)
    for i, row in enumerate(metadata.itertuples(index=False)):
        key = str(row.series_key)
        if key not in scalers:
            raise KeyError(f"No scaler found for series {key!r}")
        scaler = scalers[key]
        result[i] = float(scaled_values[i]) * float(scaler["price_std_train"]) + float(
            scaler["price_mean_train"]
        )
    return result


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if len(actual) == 0 or len(actual) != len(predicted):
        raise ValueError("Metric inputs must be non-empty and have equal lengths")
    if not (np.isfinite(actual).all() and np.isfinite(predicted).all()):
        raise ValueError("Metric inputs contain non-finite values")
    error = predicted - actual
    return {
        "n": int(len(actual)),
        "mae_rs_qtl": float(np.mean(np.abs(error))),
        "rmse_rs_qtl": float(np.sqrt(np.mean(error**2))),
        "mape_pct": float(np.mean(np.abs(error / actual)) * 100.0),
    }


def evaluate_predictions(predictions: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for crop, group in predictions.groupby("crop", sort=True):
        row: dict[str, Any] = {"crop": str(crop)}
        row.update(calculate_metrics(
            group["actual_price_rs_qtl"].to_numpy(),
            group["predicted_price_rs_qtl"].to_numpy(),
        ))
        rows.append(row)
    overall = calculate_metrics(
        predictions["actual_price_rs_qtl"].to_numpy(),
        predictions["predicted_price_rs_qtl"].to_numpy(),
    )
    return pd.DataFrame(rows), overall


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", default="data/training/monthly_maharashtra")
    parser.add_argument("--output-dir", default="data/training/monthly_maharashtra/monthly_lstm")
    parser.add_argument(
        "--baseline-predictions",
        default="data/training/monthly_maharashtra/baselines/validation_predictions.csv",
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or cuda")
    args = parser.parse_args()

    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1:
        raise ValueError("epochs, patience, and batch-size must be positive")

    training_dir = Path(args.training_dir)
    output_dir = Path(args.output_dir)
    baseline_path = Path(args.baseline_predictions)
    required_paths = [
        training_dir / "X_train.npy",
        training_dir / "y_train.npy",
        training_dir / "X_val.npy",
        training_dir / "y_val.npy",
        training_dir / "metadata_train.csv",
        training_dir / "metadata_val.csv",
        training_dir / "scalers.json",
        baseline_path,
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Required experiment files are missing:\n- " + "\n- ".join(missing))

    set_seed(args.seed)
    device = choose_device(args.device)
    x_train, y_train, x_val, y_val = load_arrays(training_dir)
    metadata_train = pd.read_csv(training_dir / "metadata_train.csv")
    metadata_val = pd.read_csv(training_dir / "metadata_val.csv")
    with (training_dir / "scalers.json").open(encoding="utf-8") as f:
        scalers: dict[str, dict[str, Any]] = json.load(f)

    required_meta = {"crop", "state", "series_key", "target_month", "target_price_rs_qtl"}
    for label, metadata, count in (
        ("train", metadata_train, len(x_train)),
        ("validation", metadata_val, len(x_val)),
    ):
        missing_columns = sorted(required_meta.difference(metadata.columns))
        if missing_columns:
            raise ValueError(f"{label} metadata is missing columns: {missing_columns}")
        if len(metadata) != count:
            raise ValueError(f"{label} metadata rows ({len(metadata)}) do not match arrays ({count})")

    # Validate that y_val is in the same order/scale as the supplied raw-price metadata.
    val_actual_from_y = inverse_scale(y_val.reshape(-1), metadata_val, scalers)
    val_actual_metadata = pd.to_numeric(metadata_val["target_price_rs_qtl"], errors="coerce").to_numpy()
    if not np.allclose(val_actual_from_y, val_actual_metadata, rtol=1e-5, atol=1e-3):
        raise ValueError(
            "Validation target array does not inverse-transform to metadata prices. "
            "Check that arrays, metadata, and scalers come from the same preprocessing run."
        )

    train_dataset = TensorDataset(
        torch.from_numpy(x_train.astype(np.float32)),
        torch.from_numpy(y_train.astype(np.float32)),
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        drop_last=False,
    )
    x_val_tensor = torch.from_numpy(x_val.astype(np.float32)).to(device)
    y_val_tensor = torch.from_numpy(y_val.astype(np.float32)).to(device)

    model = MonthlyLSTM(
        input_size=x_train.shape[2],
        hidden_size=args.hidden_size,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    loss_fn = nn.MSELoss()

    best_mae = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []

    print(f"Device: {device}")
    print(f"Train samples: {len(x_train)} | Validation samples: {len(x_val)}")
    print(f"Input shape per sample: {tuple(x_train.shape[1:])}")
    print("Test split: NOT LOADED or evaluated during this experiment")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch_x)
            loss = loss_fn(output, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += float(loss.detach().cpu()) * len(batch_x)
            seen += len(batch_x)

        model.eval()
        with torch.no_grad():
            val_scaled = model(x_val_tensor).detach().cpu().numpy().reshape(-1)
        val_predicted = inverse_scale(val_scaled, metadata_val, scalers)
        val_actual = val_actual_metadata
        val_metrics = calculate_metrics(val_actual, val_predicted)
        train_loss = running_loss / max(seen, 1)
        epoch_row: dict[str, float | int] = {
            "epoch": epoch,
            "train_mse_scaled": train_loss,
            "val_mae_rs_qtl": float(val_metrics["mae_rs_qtl"]),
            "val_rmse_rs_qtl": float(val_metrics["rmse_rs_qtl"]),
        }
        history.append(epoch_row)

        if float(val_metrics["mae_rs_qtl"]) < best_mae - 1e-6:
            best_mae = float(val_metrics["mae_rs_qtl"])
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 10 == 0 or epochs_without_improvement == 0:
            print(
                f"Epoch {epoch:03d} | train MSE={train_loss:.5f} | "
                f"val MAE=Rs.{val_metrics['mae_rs_qtl']:.2f}/qtl | "
                f"best=Rs.{best_mae:.2f} (epoch {best_epoch})"
            )
        if epochs_without_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        best_val_scaled = model(x_val_tensor).detach().cpu().numpy().reshape(-1)
    best_val_prices = inverse_scale(best_val_scaled, metadata_val, scalers)

    output_dir.mkdir(parents=True, exist_ok=True)
    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / "training_history.csv", index=False)
    torch.save(
        {
            "model_state_dict": best_state,
            "model_config": {
                "input_size": int(x_train.shape[2]),
                "hidden_size": int(args.hidden_size),
                "dropout": float(args.dropout),
                "lookback_months": int(x_train.shape[1]),
            },
            "seed": int(args.seed),
            "best_epoch": int(best_epoch),
            "best_validation_mae_rs_qtl": float(best_mae),
        },
        output_dir / "best_model.pt",
    )

    val_predictions = metadata_val[["crop", "state", "series_key", "target_month"]].copy()
    val_predictions["actual_price_rs_qtl"] = val_actual_metadata
    val_predictions["predicted_price_rs_qtl"] = best_val_prices
    val_predictions.to_csv(output_dir / "validation_predictions.csv", index=False)
    by_crop, overall = evaluate_predictions(val_predictions)
    by_crop.to_csv(output_dir / "validation_metrics_by_crop.csv", index=False)

    baseline = pd.read_csv(baseline_path)
    baseline_required = {
        "crop", "state", "target_month", "baseline", "actual_price_rs_qtl", "predicted_price_rs_qtl"
    }
    if missing_columns := sorted(baseline_required.difference(baseline.columns)):
        raise ValueError(f"Baseline predictions are missing columns: {missing_columns}")
    baseline = baseline.loc[baseline["baseline"] == "last_month_naive"].copy()
    keys = ["crop", "state", "target_month"]
    comparison = val_predictions.merge(
        baseline[keys + ["actual_price_rs_qtl", "predicted_price_rs_qtl"]].rename(
            columns={
                "actual_price_rs_qtl": "baseline_actual_price_rs_qtl",
                "predicted_price_rs_qtl": "last_month_naive_prediction_rs_qtl",
            }
        ),
        on=keys,
        how="left",
        validate="one_to_one",
    )
    if comparison["last_month_naive_prediction_rs_qtl"].isna().any():
        raise ValueError("Could not match all validation rows to last-month baseline predictions")
    if not np.allclose(
        comparison["actual_price_rs_qtl"],
        comparison["baseline_actual_price_rs_qtl"],
        rtol=1e-7,
        atol=1e-5,
    ):
        raise ValueError("Baseline and LSTM validation actual prices do not match")
    comparison = comparison.drop(columns=["baseline_actual_price_rs_qtl"])
    comparison.to_csv(output_dir / "validation_baseline_comparison.csv", index=False)

    baseline_overall = calculate_metrics(
        comparison["actual_price_rs_qtl"].to_numpy(),
        comparison["last_month_naive_prediction_rs_qtl"].to_numpy(),
    )
    comparison_by_crop_rows: list[dict[str, Any]] = []
    for crop, group in comparison.groupby("crop", sort=True):
        lstm_m = calculate_metrics(
            group["actual_price_rs_qtl"].to_numpy(),
            group["predicted_price_rs_qtl"].to_numpy(),
        )
        naive_m = calculate_metrics(
            group["actual_price_rs_qtl"].to_numpy(),
            group["last_month_naive_prediction_rs_qtl"].to_numpy(),
        )
        comparison_by_crop_rows.append({
            "crop": str(crop),
            "n": int(lstm_m["n"]),
            "lstm_mae_rs_qtl": float(lstm_m["mae_rs_qtl"]),
            "last_month_naive_mae_rs_qtl": float(naive_m["mae_rs_qtl"]),
            "lstm_rmse_rs_qtl": float(lstm_m["rmse_rs_qtl"]),
            "last_month_naive_rmse_rs_qtl": float(naive_m["rmse_rs_qtl"]),
            "lstm_mape_pct": float(lstm_m["mape_pct"]),
            "last_month_naive_mape_pct": float(naive_m["mape_pct"]),
        })
    comparison_by_crop = pd.DataFrame(comparison_by_crop_rows)
    comparison_by_crop.to_csv(output_dir / "validation_comparison_by_crop.csv", index=False)

    summary = {
        "experiment": "maharashtra_monthly_shared_lstm",
        "test_split_used": False,
        "selection_metric": "validation MAE in Rs./quintal",
        "best_epoch": int(best_epoch),
        "epochs_completed": int(len(history)),
        "device": str(device),
        "seed": int(args.seed),
        "model_config": {
            "input_size": int(x_train.shape[2]),
            "lookback_months": int(x_train.shape[1]),
            "hidden_size": int(args.hidden_size),
            "dropout": float(args.dropout),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
        },
        "lstm_validation_overall": overall,
        "last_month_naive_validation_overall": baseline_overall,
        "lstm_beats_last_month_naive_by_mae": bool(
            overall["mae_rs_qtl"] < baseline_overall["mae_rs_qtl"]
        ),
        "per_crop_validation": comparison_by_crop_rows,
    }
    with (output_dir / "experiment_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n--- MONTHLY LSTM VALIDATION METRICS (Rs./quintal) ---")
    print(by_crop.to_string(index=False, float_format=lambda value: f"{value:,.2f}"))
    print(
        f"Overall: n={overall['n']}, MAE={overall['mae_rs_qtl']:.2f}, "
        f"RMSE={overall['rmse_rs_qtl']:.2f}, MAPE={overall['mape_pct']:.2f}%"
    )
    print("\n--- LSTM vs LAST-MONTH BASELINE ---")
    print(comparison_by_crop.to_string(index=False, float_format=lambda value: f"{value:,.2f}"))
    print(
        f"\nLast-month baseline MAE={baseline_overall['mae_rs_qtl']:.2f}; "
        f"LSTM MAE={overall['mae_rs_qtl']:.2f}"
    )
    print(f"Saved model and validation outputs to: {output_dir}")
    print("Test split remains untouched. Do not use test results for model tuning.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
