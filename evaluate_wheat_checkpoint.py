"""
Read-only exploratory backtest for the saved wheat LSTM checkpoint.

Does not train the model or modify model/scaler/history artifacts.
Forecast horizon means the next 30 observed records, not necessarily
30 consecutive calendar days.
"""

from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from config import LSTM_CONFIG
from fetcher import MandiPriceFetcher, WeatherFetcher
from preprocessor import merge_datasets, apply_scalers, load_scalers
from lstm_model import CropPriceLSTM, get_device


ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw" / "agmarknet"
CHECKPOINT = ROOT / "models" / "saved" / "best_wheat.pt"
SCALER_DIR = ROOT / "models" / "scalers"
REPORT_DIR = ROOT / "reports"

CROP = "wheat"
STATE = "Punjab"


def metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float).ravel()
    predicted = np.asarray(predicted, dtype=float).ravel()

    errors = predicted - actual
    nonzero = np.abs(actual) > 1e-12

    return {
        "MAE_Rs_per_quintal": float(np.mean(np.abs(errors))),
        "RMSE_Rs_per_quintal": float(np.sqrt(np.mean(errors ** 2))),
        "MAPE_percent": (
            float(np.mean(np.abs(errors[nonzero] / actual[nonzero])) * 100)
            if nonzero.any()
            else float("nan")
        ),
    }


def load_cached_prices():
    files = sorted(RAW_DIR.glob("wheat_punjab_*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No cached wheat/Punjab CSVs found in {RAW_DIR}"
        )

    frames = []
    for path in files:
        frame = pd.read_csv(path)
        if "date" not in frame.columns:
            raise ValueError(f"'date' column missing from {path}")
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frames.append(frame)

    market_df = pd.concat(frames, ignore_index=True)
    market_df = market_df.dropna(subset=["date"])

    required = {
        "state", "market", "arrivals_tonnes",
        "min_price", "max_price", "modal_price",
    }
    missing = required - set(market_df.columns)
    if missing:
        raise ValueError(
            f"Raw AGMARKNET data is missing required columns: {sorted(missing)}"
        )

    for col in ["arrivals_tonnes", "min_price", "max_price", "modal_price"]:
        market_df[col] = pd.to_numeric(market_df[col], errors="coerce")

    market_df = market_df.dropna(
        subset=["arrivals_tonnes", "min_price", "max_price", "modal_price"]
    )

    # Reuse the project's existing daily aggregation logic.
    fetcher = MandiPriceFetcher()
    daily = fetcher._aggregate_agmarknet_daily(market_df)

    if daily.empty:
        raise ValueError("Daily aggregation produced no records.")

    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").reset_index(drop=True)

    return daily, len(files), len(market_df)


def main():
    # Fail closed: never silently evaluate randomly initialized weights.
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    scaler_path = SCALER_DIR / "scalers_wheat.pkl"
    if not scaler_path.is_file():
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

    feature_cols = list(LSTM_CONFIG.feature_columns)
    target_col = LSTM_CONFIG.target_column
    seq_len = LSTM_CONFIG.sequence_length
    horizon = LSTM_CONFIG.forecast_horizon

    print("Loading cached AGMARKNET records...")
    prices, file_count, market_row_count = load_cached_prices()

    start_date = prices["date"].min().strftime("%Y-%m-%d")
    end_date = prices["date"].max().strftime("%Y-%m-%d")

    print(f"Monthly cache files: {file_count}")
    print(f"Market-level rows after basic cleaning: {market_row_count}")
    print(f"Aggregated price observations: {len(prices)}")
    print(f"Observed period: {start_date} to {end_date}")

    gaps = prices["date"].sort_values().diff().dt.days.dropna()
    if not gaps.empty:
        print(f"Median gap between observed dates: {gaps.median():.1f} days")
        print(f"Largest gap between observed dates: {gaps.max()} days")

    print("\nFetching historical weather for this period...")
    weather = WeatherFetcher().fetch(STATE, start_date, end_date)
    weather_source = weather.attrs.get("data_source", "unknown")
    print(f"Weather source reported by fetcher: {weather_source}")

    # Uses the project's existing feature engineering. This also drops
    # early rows needed to create lag features, as in the training pipeline.
    prepared = merge_datasets(prices, weather)
    prepared = prepared.sort_values("date").reset_index(drop=True)

    missing_features = set(feature_cols) - set(prepared.columns)
    if missing_features:
        raise ValueError(
            f"Prepared data is missing model features: {sorted(missing_features)}"
        )

    if target_col not in feature_cols:
        raise ValueError(
            f"Target {target_col!r} must be included in feature_cols."
        )

    # Load the original fitted scalers; do not fit or save new ones.
    scalers = load_scalers(CROP)
    missing_scalers = set(feature_cols) - set(scalers)
    if missing_scalers:
        raise ValueError(
            f"Saved scaler file is missing: {sorted(missing_scalers)}"
        )

    scaled = apply_scalers(prepared, scalers)
    data = scaled[feature_cols].to_numpy(dtype=np.float32)
    target_idx = feature_cols.index(target_col)
    target_scaler = scalers[target_col]

    n_windows = len(data) - seq_len - horizon + 1
    if n_windows < 1:
        raise ValueError(
            f"Not enough prepared observations ({len(data)}) for "
            f"sequence_length={seq_len} and horizon={horizon}. "
            "No metrics were calculated."
        )

    # Explicitly load the saved checkpoint and validate its state dict.
    device = get_device()
    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=True,
    )
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise ValueError(
            "Checkpoint does not contain 'model_state'; refusing to evaluate."
        )

    model = CropPriceLSTM(n_features=len(feature_cols)).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()

    print(f"\nCheckpoint: {CHECKPOINT}")
    print(f"Checkpoint epoch metadata: {checkpoint.get('epoch', 'not recorded')}")
    print(f"Prepared observations after feature engineering: {len(prepared)}")
    print(f"Rolling forecast windows: {n_windows}")
    print("Running inference only; no training will be performed...")

    # Build rolling windows in memory. Each prediction is compared with the
    # next `horizon` observed records, which may have irregular date spacing.
    X, actual_scaled, origins = [], [], []
    for i in range(n_windows):
        X.append(data[i : i + seq_len])
        actual_scaled.append(
            data[i + seq_len : i + seq_len + horizon, target_idx]
        )
        origins.append(i + seq_len - 1)

    X_tensor = torch.tensor(np.asarray(X), dtype=torch.float32, device=device)

    with torch.no_grad():
        predicted_scaled = model(X_tensor).detach().cpu().numpy()

    if predicted_scaled.shape != (n_windows, horizon):
        raise ValueError(
            f"Unexpected model output shape: {predicted_scaled.shape}"
        )

    actual_scaled = np.asarray(actual_scaled, dtype=float)

    # Convert model output and actual targets to original price units.
    actual = target_scaler.inverse_transform(
        actual_scaled.reshape(-1, 1)
    ).reshape(n_windows, horizon)

    predicted = target_scaler.inverse_transform(
        predicted_scaled.reshape(-1, 1)
    ).reshape(n_windows, horizon)

    # Persistence baseline: repeat the last observed modal price.
    last_scaled = np.array(
        [data[i, target_idx] for i in origins], dtype=float
    ).reshape(-1, 1)
    last_price = target_scaler.inverse_transform(last_scaled).reshape(-1)
    persistence = np.repeat(last_price[:, None], horizon, axis=1)

    print("\n=== Exploratory rolling backtest ===")
    print("All prices below are in original units (₹ per quintal).")
    print("Metrics pool overlapping forecast windows; they are not independent.")
    print("\nLSTM:")
    for key, value in metrics(actual, predicted).items():
        print(f"  {key}: {value:.4f}")

    print("\nLast-observed-price baseline:")
    for key, value in metrics(actual, persistence).items():
        print(f"  {key}: {value:.4f}")

    # Save a detailed, separate report with one row per forecasted target.
    rows = []
    dates = prepared["date"].reset_index(drop=True)
    for i in range(n_windows):
        origin_idx = origins[i]
        for step in range(horizon):
            target_idx_row = origin_idx + 1 + step
            rows.append({
                "forecast_origin_date": dates.iloc[origin_idx].date().isoformat(),
                "target_date": dates.iloc[target_idx_row].date().isoformat(),
                "horizon_observations": step + 1,
                "actual_price_rs_per_quintal": float(actual[i, step]),
                "lstm_prediction_rs_per_quintal": float(predicted[i, step]),
                "persistence_prediction_rs_per_quintal": float(persistence[i, step]),
                "lstm_absolute_error": float(abs(predicted[i, step] - actual[i, step])),
                "persistence_absolute_error": float(abs(persistence[i, step] - actual[i, step])),
            })

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORT_DIR / f"wheat_checkpoint_backtest_{timestamp}.csv"
    pd.DataFrame(rows).to_csv(report_path, index=False)

    print(f"\nDetailed predictions saved to: {report_path}")
    print("\nIMPORTANT:")
    print("- This is an exploratory historical backtest, not an independent test.")
    print("- The checkpoint's original training-data provenance is unconfirmed.")
    print("- The 30-step horizon counts observed records, not calendar days.")
    print("- Weather may be climate normals if historical retrieval failed.")
    print("- No model, scaler, or training-history artifact was overwritten.")


if __name__ == "__main__":
    main()
