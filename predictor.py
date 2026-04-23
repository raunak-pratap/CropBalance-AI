"""
models/predictor.py
-------------------
Loads a trained LSTM checkpoint and runs inference.
Used by the FastAPI server to serve predictions.
"""

import os
import numpy as np
import pandas as pd
import torch
from loguru import logger
from typing import List, Dict
from datetime import datetime, timedelta

from config import LSTM_CONFIG, PATH_CONFIG
from lstm_model import CropPriceLSTM, get_device
from preprocessor import load_scalers, apply_scalers, add_cyclical_features, add_lag_features


class CropPredictor:
    """
    Loads trained model + scalers and exposes a predict() method.

    Usage:
        predictor = CropPredictor(crop="wheat")
        result    = predictor.predict(recent_df)
    """

    def __init__(self, crop: str, n_features: int = None):
        self.crop    = crop
        self.device  = get_device()
        self.scalers = load_scalers(crop)

        # Infer feature count from scalers
        self.feature_cols = list(self.scalers.keys())
        n_features = n_features or len(self.feature_cols)

        # Build and load model
        self.model = CropPriceLSTM(n_features=n_features).to(self.device)
        self._load_checkpoint()
        self.model.eval()

    def predict(self, recent_df: pd.DataFrame) -> Dict:
        """
        Given a DataFrame of recent observations (>= seq_len rows),
        returns a multi-step price forecast for the next `horizon` days.

        Args:
            recent_df: DataFrame with all raw feature columns + 'date'

        Returns:
            {
                "crop":         str,
                "forecast":     list[dict],   # date + price_inr
                "generated_at": str (ISO),
            }
        """
        cfg = LSTM_CONFIG
        seq_len = cfg.sequence_length
        horizon = cfg.forecast_horizon

        # Feature engineering (same as training)
        df = recent_df.copy().sort_values("date").reset_index(drop=True)
        df = add_cyclical_features(df)
        df = add_lag_features(df)
        df = df.dropna().reset_index(drop=True)

        if len(df) < seq_len:
            raise ValueError(
                f"Need at least {seq_len} rows of history, got {len(df)}."
            )

        # Scale using saved scalers
        df_scaled = apply_scalers(df, self.scalers)

        # Take last seq_len rows as input window
        available_cols = [c for c in self.feature_cols if c in df_scaled.columns]
        window = df_scaled[available_cols].values[-seq_len:]   # (seq_len, n_features)

        X = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            scaled_preds = self.model(X).cpu().numpy().flatten()  # (horizon,)

        # Inverse-transform to ₹/quintal
        target_scaler = self.scalers[cfg.target_column]
        prices_inr = target_scaler.inverse_transform(
            scaled_preds.reshape(-1, 1)
        ).flatten()

        # Build date range for forecast
        last_date   = df["date"].max()
        future_dates = [last_date + timedelta(days=i + 1) for i in range(horizon)]

        forecast = [
            {
                "date":      d.strftime("%Y-%m-%d"),
                "price_inr": round(float(p), 2),
            }
            for d, p in zip(future_dates, prices_inr)
        ]

        return {
            "crop":         self.crop,
            "forecast":     forecast,
            "generated_at": datetime.utcnow().isoformat(),
        }

    def _load_checkpoint(self):
        ckpt_path = os.path.join(PATH_CONFIG.models_dir, f"best_{self.crop}.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"No trained model found at {ckpt_path}. "
                f"Run train.py for crop='{self.crop}' first."
            )
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        logger.info(f"Model loaded ← {ckpt_path} (epoch {ckpt['epoch']})")
