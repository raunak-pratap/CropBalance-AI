"""
predictor.py
------------
Loads a trained LSTM checkpoint and runs inference.
Used by the FastAPI server to serve predictions.
"""

import os
from datetime import datetime, timedelta
from typing import Dict

import pandas as pd
import torch
from loguru import logger

from config import LSTM_CONFIG, PATH_CONFIG
from lstm_model import CropPriceLSTM, get_device
from preprocessor import (
    load_scalers,
    apply_scalers,
    add_cyclical_features,
    add_lag_features,
)


class CropPredictor:
    """
    Loads a trained model and performs crop price prediction.

    Example:
        predictor = CropPredictor(crop="wheat")
        result = predictor.predict(recent_df)
    """

    def __init__(self, crop: str, n_features: int = None):
        self.crop = crop
        self.device = get_device()

        # Load saved scalers
        self.scalers = load_scalers(crop)

        # Infer feature columns
        self.feature_cols = list(self.scalers.keys())
        if n_features is None:
            n_features = len(self.feature_cols)

        # Build model
        self.model = CropPriceLSTM(
            n_features=n_features
        ).to(self.device)

        self._load_checkpoint()
        self.model.eval()

    def predict(self, recent_df: pd.DataFrame) -> Dict:
        """
        Predict future crop prices.

        Parameters
        ----------
        recent_df : pandas.DataFrame

        Returns
        -------
        dict
        """

        cfg = LSTM_CONFIG

        seq_len = cfg.sequence_length
        horizon = cfg.forecast_horizon

        df = recent_df.copy()
        df = df.sort_values("date").reset_index(drop=True)

        # Feature engineering
        df = add_cyclical_features(df)
        df = add_lag_features(df)

        df = df.dropna().reset_index(drop=True)

        if len(df) < seq_len:
            raise ValueError(
                f"Need at least {seq_len} rows of history, got {len(df)}."
            )

        # Scale
        df_scaled = apply_scalers(df, self.scalers)

        available_cols = [
            c for c in self.feature_cols
            if c in df_scaled.columns
        ]

        window = df_scaled[available_cols].values[-seq_len:]

        X = (
            torch.tensor(
                window,
                dtype=torch.float32,
            )
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.no_grad():
            scaled_preds = (
                self.model(X)
                .cpu()
                .numpy()
                .flatten()
            )

        # Inverse scaling
        target_scaler = self.scalers[cfg.target_column]

        prices = target_scaler.inverse_transform(
            scaled_preds.reshape(-1, 1)
        ).flatten()

        last_date = df["date"].max()

        future_dates = [
            last_date + timedelta(days=i + 1)
            for i in range(horizon)
        ]

        forecast = []

        for d, p in zip(future_dates, prices):
            forecast.append(
                {
                    "date": d.strftime("%Y-%m-%d"),
                    "price_inr": round(float(p), 2),
                }
            )

        return {
            "crop": self.crop,
            "forecast": forecast,
            "generated_at": datetime.utcnow().isoformat(),
        }

    def _load_checkpoint(self):
        """
        Load trained model checkpoint.
        """

        ckpt_path = os.path.join(
            PATH_CONFIG.models_dir,
            f"best_{self.crop}.pt",
        )

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"No trained model found at {ckpt_path}. "
                f"Run train.py for crop='{self.crop}' first."
            )

        checkpoint = torch.load(
            ckpt_path,
            map_location=self.device,
        )

        self.model.load_state_dict(
            checkpoint["model_state"]
        )

        logger.info(
            f"Model loaded <- {ckpt_path} "
            f"(epoch {checkpoint.get('epoch', '?')})"
        )