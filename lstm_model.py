"""
models/lstm_model.py
--------------------
Multi-layer stacked LSTM for multi-step crop price forecasting.

Architecture:
  Input  → LSTM (stacked, with dropout) → FC head → Output
  Input shape:  (batch, seq_len, n_features)
  Output shape: (batch, forecast_horizon)
"""

import torch
import torch.nn as nn
from loguru import logger
from config import LSTMConfig, LSTM_CONFIG


class CropPriceLSTM(nn.Module):
    """
    Stacked LSTM with:
    - Layer normalisation after each LSTM layer (stabilises training)
    - Dropout between layers (prevents overfitting)
    - Two-layer FC head with ReLU (captures non-linear price dynamics)
    - Xavier weight initialisation
    """

    def __init__(self, n_features: int, cfg: LSTMConfig = LSTM_CONFIG):
        super().__init__()
        self.cfg         = cfg
        self.hidden_size = cfg.hidden_size
        self.num_layers  = cfg.num_layers

        # ── LSTM layers ───────────────────────
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=cfg.bidirectional,
        )

        lstm_out_size = cfg.hidden_size * (2 if cfg.bidirectional else 1)

        # ── Layer norm on LSTM output ──────────
        self.layer_norm = nn.LayerNorm(lstm_out_size)

        # ── FC prediction head ─────────────────
        self.fc = nn.Sequential(
            nn.Linear(lstm_out_size, cfg.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_size // 2, cfg.forecast_horizon),
        )

        self._init_weights()
        logger.info(
            f"CropPriceLSTM | features={n_features} | "
            f"hidden={cfg.hidden_size} | layers={cfg.num_layers} | "
            f"horizon={cfg.forecast_horizon}d | "
            f"params={self.count_parameters():,}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, n_features)
        returns: (batch, forecast_horizon)  — scaled price predictions
        """
        # lstm_out: (batch, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x)

        # Take the output at the last time step
        last_step = lstm_out[:, -1, :]          # (batch, hidden_size)
        normed    = self.layer_norm(last_step)   # Stabilise
        out       = self.fc(normed)              # (batch, forecast_horizon)
        return out

    def _init_weights(self):
        """Xavier uniform for FC layers; orthogonal for LSTM weights."""
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)
                # Set forget gate bias to 1 (helps remember long sequences)
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1)
            elif "weight" in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param.data)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ──────────────────────────────────────────────
# Convenience builders
# ──────────────────────────────────────────────

def build_model(n_features: int, cfg: LSTMConfig = LSTM_CONFIG) -> CropPriceLSTM:
    """Builds model and moves to best available device."""
    device = get_device()
    model  = CropPriceLSTM(n_features=n_features, cfg=cfg).to(device)
    return model


def get_device() -> torch.device:
    if torch.cuda.is_available():
        logger.info("Using GPU (CUDA)")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        logger.info("Using Apple MPS")
        return torch.device("mps")
    else:
        logger.info("Using CPU")
        return torch.device("cpu")
