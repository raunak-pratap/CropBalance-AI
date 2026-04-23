"""
data/preprocessor.py
--------------------
Merges mandi price and weather data, engineers time-aware features,
and prepares PyTorch-ready sliding-window sequences for LSTM training.
"""

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import torch
from loguru import logger
from typing import Tuple, Dict, Optional

from config import LSTM_CONFIG, PATH_CONFIG


# ──────────────────────────────────────────────
# Feature Engineering
# ──────────────────────────────────────────────

def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode month and week as sine/cosine so the model understands cyclicality."""
    df["month_sin"] = np.sin(2 * np.pi * df["date"].dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["date"].dt.month / 12)
    df["week_sin"]  = np.sin(2 * np.pi * df["date"].dt.isocalendar().week.astype(float) / 52)
    df["week_cos"]  = np.cos(2 * np.pi * df["date"].dt.isocalendar().week.astype(float) / 52)
    return df


def add_lag_features(df: pd.DataFrame, target: str = "modal_price") -> pd.DataFrame:
    """Rolling statistics and lag features for the price column."""
    for lag in [1, 7, 14, 30]:
        df[f"price_lag_{lag}d"] = df[target].shift(lag)
    df["price_roll7_mean"]  = df[target].rolling(7).mean()
    df["price_roll30_mean"] = df[target].rolling(30).mean()
    df["price_roll7_std"]   = df[target].rolling(7).std()
    df["price_pct_change"]  = df[target].pct_change()
    return df


def merge_datasets(
    price_df: pd.DataFrame,
    weather_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merges price and weather dataframes on (date, state),
    fills gaps, and returns a clean daily time series.
    """
    df = pd.merge(price_df, weather_df, on=["date", "state"], how="left")

    # Forward-fill weather gaps (weekend / holiday gaps)
    weather_cols = ["temp_max", "temp_min", "rainfall_mm", "humidity_pct"]
    df[weather_cols] = df[weather_cols].ffill().bfill()

    # Sort chronologically
    df = df.sort_values("date").reset_index(drop=True)

    # Add engineered features
    df = add_cyclical_features(df)
    df = add_lag_features(df)

    # Drop rows that have NaNs from lag creation
    df = df.dropna().reset_index(drop=True)

    logger.info(f"Merged dataset: {len(df)} rows, {df.shape[1]} columns")
    return df


# ──────────────────────────────────────────────
# Scaler management
# ──────────────────────────────────────────────

def fit_scalers(df: pd.DataFrame, feature_cols: list) -> Dict[str, MinMaxScaler]:
    """Fits a scaler per feature column (important: target has its own scaler)."""
    scalers = {}
    for col in feature_cols:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(df[[col]])
        scalers[col] = scaler
    return scalers


def apply_scalers(df: pd.DataFrame, scalers: Dict[str, MinMaxScaler]) -> pd.DataFrame:
    df = df.copy()
    for col, scaler in scalers.items():
        if col in df.columns:
            df[col] = scaler.transform(df[[col]])
    return df


def save_scalers(scalers: Dict[str, MinMaxScaler], crop: str) -> str:
    os.makedirs(PATH_CONFIG.scalers_dir, exist_ok=True)
    path = os.path.join(PATH_CONFIG.scalers_dir, f"scalers_{crop}.pkl")
    joblib.dump(scalers, path)
    logger.info(f"Scalers saved → {path}")
    return path


def load_scalers(crop: str) -> Dict[str, MinMaxScaler]:
    path = os.path.join(PATH_CONFIG.scalers_dir, f"scalers_{crop}.pkl")
    scalers = joblib.load(path)
    logger.info(f"Scalers loaded ← {path}")
    return scalers


# ──────────────────────────────────────────────
# Sliding-window sequence builder
# ──────────────────────────────────────────────

def build_sequences(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    seq_len: int,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts a scaled DataFrame into overlapping (X, y) windows.

    X shape: (num_samples, seq_len, num_features)
    y shape: (num_samples, horizon)
    """
    data = df[feature_cols].values
    target_idx = feature_cols.index(target_col)

    X, y = [], []
    for i in range(len(data) - seq_len - horizon + 1):
        X.append(data[i : i + seq_len])
        # Multi-step target: next `horizon` days of price
        y.append(data[i + seq_len : i + seq_len + horizon, target_idx])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ──────────────────────────────────────────────
# PyTorch Dataset
# ──────────────────────────────────────────────

class CropPriceDataset(Dataset):
    """Wraps numpy sequences as a PyTorch Dataset."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def make_dataloaders(
    df_scaled: pd.DataFrame,
    cfg=LSTM_CONFIG,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Builds train / val / test DataLoaders from the scaled DataFrame.
    Uses chronological (non-shuffled) split to avoid data leakage.
    """
    X, y = build_sequences(
        df_scaled,
        cfg.feature_columns,
        cfg.target_column,
        cfg.sequence_length,
        cfg.forecast_horizon,
    )

    n = len(X)
    n_train = int(n * cfg.train_split)
    n_val   = int(n * cfg.val_split)

    X_train, y_train = X[:n_train],          y[:n_train]
    X_val,   y_val   = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test,  y_test  = X[n_train+n_val:],    y[n_train+n_val:]

    logger.info(f"Sequences → train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")

    train_ds = CropPriceDataset(X_train, y_train)
    val_ds   = CropPriceDataset(X_val,   y_val)
    test_ds  = CropPriceDataset(X_test,  y_test)

    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=False)
    val_dl   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False)
    test_dl  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False)

    return train_dl, val_dl, test_dl


# ──────────────────────────────────────────────
# Full preprocessing pipeline (one call)
# ──────────────────────────────────────────────

def run_preprocessing_pipeline(
    price_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    crop: str,
    save: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    End-to-end: merge → engineer → scale → split → DataLoaders.

    Returns (train_dl, val_dl, test_dl, meta) where meta contains
    the scalers, feature columns, and merged DataFrame (for analysis).
    """
    cfg = LSTM_CONFIG

    # 1. Merge & engineer features
    df = merge_datasets(price_df, weather_df)

    # 2. Keep only the columns the model needs (some lag cols may not be in cfg)
    available = [c for c in cfg.feature_columns if c in df.columns]
    if len(available) < len(cfg.feature_columns):
        missing = set(cfg.feature_columns) - set(available)
        logger.warning(f"Missing feature columns (will skip): {missing}")

    # 3. Fit scalers on training portion only (prevent leakage)
    n_train = int(len(df) * cfg.train_split)
    train_slice = df.iloc[:n_train]
    scalers = fit_scalers(train_slice, available)
    if save:
        save_scalers(scalers, crop)

    # 4. Scale full dataset
    df_scaled = apply_scalers(df, scalers)

    # 5. Build DataLoaders
    cfg_copy        = LSTM_CONFIG
    cfg_copy.feature_columns = available   # Use only available columns
    train_dl, val_dl, test_dl = make_dataloaders(df_scaled, cfg_copy)

    meta = {
        "scalers":         scalers,
        "feature_columns": available,
        "df_merged":       df,
        "df_scaled":       df_scaled,
        "n_features":      len(available),
    }

    return train_dl, val_dl, test_dl, meta
