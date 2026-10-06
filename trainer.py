"""
models/trainer.py
-----------------
Training loop with:
- Early stopping (patience-based)
- ReduceLROnPlateau scheduler
- Comprehensive metrics (MAE, RMSE, MAPE)
- Checkpointing (saves best model automatically)
- Training curve logging
"""

import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from loguru import logger
from typing import Dict, Tuple, List

from config import LSTM_CONFIG, PATH_CONFIG
from lstm_model import CropPriceLSTM, get_device


# ──────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae":  round(mae(y_true, y_pred), 4),
        "rmse": round(rmse(y_true, y_pred), 4),
        "mape": round(mape(y_true, y_pred), 4),
    }


# ──────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────

class Trainer:
    def __init__(
        self,
        model: CropPriceLSTM,
        crop: str,
        cfg=LSTM_CONFIG,
    ):
        self.model  = model
        self.crop   = crop
        self.cfg    = cfg
        self.device = get_device()

        os.makedirs(PATH_CONFIG.models_dir, exist_ok=True)
        self.ckpt_path = os.path.join(PATH_CONFIG.models_dir, f"best_{crop}.pt")

        self.optimizer = AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5,
            patience=5, min_lr=1e-6,
        )
        self.criterion = nn.HuberLoss(delta=1.0)  # Robust to outliers vs plain MSE

        self.history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [],
            "val_mae":    [], "val_rmse": [], "val_mape": [],
        }

    # ── One epoch ─────────────────────────────
    def _run_epoch(self, loader: DataLoader, train: bool) -> Tuple[float, np.ndarray, np.ndarray]:
        self.model.train(train)
        total_loss = 0.0
        all_preds, all_targets = [], []

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                preds = self.model(X_batch)
                loss  = self.criterion(preds, y_batch)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                total_loss  += loss.item() * len(X_batch)
                all_preds.append(preds.detach().cpu().numpy())
                all_targets.append(y_batch.detach().cpu().numpy())
        avg_loss = total_loss / len(loader.dataset)
        return avg_loss, np.concatenate(all_preds), np.concatenate(all_targets)

    # ── Full training loop ─────────────────────
    def train(
        self,
        train_dl: DataLoader,
        val_dl: DataLoader,
    ) -> Dict:
        best_val_loss = float("inf")
        patience_ctr  = 0
        t0            = time.time()

        logger.info(f"Training started: {self.cfg.epochs} epochs | crop={self.crop}")
        logger.info("-" * 60)

        for epoch in range(1, self.cfg.epochs + 1):
            train_loss, _, _   = self._run_epoch(train_dl, train=True)
            val_loss, preds, targets = self._run_epoch(val_dl, train=False)

            self.scheduler.step(val_loss)
            metrics = compute_metrics(targets.flatten(), preds.flatten())

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_mae"].append(metrics["mae"])
            self.history["val_rmse"].append(metrics["rmse"])
            self.history["val_mape"].append(metrics["mape"])

            elapsed = time.time() - t0
            lr = self.optimizer.param_groups[0]["lr"]

            logger.info(
                f"Epoch {epoch:03d}/{self.cfg.epochs} | "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"MAE={metrics['mae']:.4f} | MAPE={metrics['mape']:.2f}% | "
                f"lr={lr:.2e} | {elapsed:.0f}s"
            )

            # Checkpoint best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_ctr  = 0
                self._save_checkpoint(epoch, val_loss, metrics)
            else:
                patience_ctr += 1
                if patience_ctr >= self.cfg.patience:
                    logger.info(f"Early stopping at epoch {epoch} (patience={self.cfg.patience})")
                    break

        total_time = time.time() - t0
        logger.info(f"Training complete in {total_time:.1f}s | best val_loss={best_val_loss:.4f}")

        # Save training history
        hist_path = os.path.join(PATH_CONFIG.models_dir, f"history_{self.crop}.json")
        with open(hist_path, "w") as f:
            json.dump(self.history, f, indent=2)

        return self.history

    def evaluate(
        self,
        test_dl: DataLoader,
        target_scaler=None,
    ) -> Dict[str, float]:
        """Evaluate on test set using best saved checkpoint."""

        self._load_checkpoint()
        _, preds, targets = self._run_epoch(test_dl, train=False)

        # Convert scaled predictions/targets back to original price units
        if target_scaler is not None:
            original_shape = preds.shape

            preds = target_scaler.inverse_transform(
                preds.reshape(-1, 1)
            ).reshape(original_shape)

            targets = target_scaler.inverse_transform(
                targets.reshape(-1, 1)
            ).reshape(original_shape)

        metrics = compute_metrics(
            targets.flatten(),
            preds.flatten()
        )

        logger.info(f"Test metrics: {metrics}")
        return metrics

    def evaluate_naive_baseline(
        self,
        test_dl: DataLoader,
        target_scaler=None,
        target_index: int = 0,
    ) -> Dict[str, float]:
        """
        Naive persistence baseline:
        predict every future day using the last observed target price.
        """

        all_preds = []
        all_targets = []

        for X_batch, y_batch in test_dl:
            # Use the actual target feature index instead of assuming index 0.
            last_price_scaled = X_batch[:, -1, target_index]

            horizon = y_batch.shape[1]

            baseline = last_price_scaled.unsqueeze(1).repeat(1, horizon)

            all_preds.append(baseline.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())

        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)

        if target_scaler is not None:
            original_shape = preds.shape

            preds = target_scaler.inverse_transform(
                preds.reshape(-1, 1)
            ).reshape(original_shape)

            targets = target_scaler.inverse_transform(
                targets.reshape(-1, 1)
            ).reshape(original_shape)

        metrics = compute_metrics(
            targets.flatten(),
            preds.flatten()
        )

        logger.info(f"Naive baseline metrics: {metrics}")

        return metrics

    def evaluate_seasonal_naive_baseline(
        self,
        test_dl: DataLoader,
        target_scaler=None,
        target_index: int = 0,
        season_length: int = 7,
    ) -> Dict[str, float]:
        """
        Evaluate a true horizon-wise seasonal naive baseline.

        For a 7-day season:
            Day 1 forecast -> price from 7 days ago
            Day 2 forecast -> price from 6 days ago
            ...
            Day 7 forecast -> latest observed price
            Day 8 forecast -> repeat Day 1 seasonal position
            ...
        """

        if season_length > self.cfg.sequence_length:
            raise ValueError(
                f"season_length ({season_length}) cannot exceed "
                f"sequence_length ({self.cfg.sequence_length})"
            )

        all_preds = []
        all_targets = []

        for X_batch, y_batch in test_dl:

            # Last 7 observed target values.
            # Shape: [batch, 7]
            seasonal_history = X_batch[
                :, -season_length:, target_index
            ]

            horizon = y_batch.shape[1]

            # Horizon-wise seasonal naive prediction.
            #
            # h=0 -> oldest value in the 7-day window
            # h=1 -> next value
            # ...
            # h=6 -> latest observed value
            # h=7 -> repeat oldest value
            seasonal_preds = torch.stack(
                [
                    seasonal_history[:, h % season_length]
                    for h in range(horizon)
                ],
                dim=1,
            )

            all_preds.append(seasonal_preds.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())

        preds = np.concatenate(all_preds, axis=0)
        targets = np.concatenate(all_targets, axis=0)

        # Convert scaled values back to INR.
        if target_scaler is not None:
            preds = target_scaler.inverse_transform(
                preds.reshape(-1, 1)
            ).reshape(preds.shape)

            targets = target_scaler.inverse_transform(
                targets.reshape(-1, 1)
            ).reshape(targets.shape)

        metrics = compute_metrics(
            targets.flatten(),
            preds.flatten(),
        )

        logger.info(
            f"Seasonal naive ({season_length}d) | "
            f"MAE: ₹{metrics['mae']:.2f} | "
            f"RMSE: ₹{metrics['rmse']:.2f} | "
            f"MAPE: {metrics['mape']:.2f}%"
        )

        return metrics

    def evaluate_horizons(
        self,
        test_dl: DataLoader,
        target_scaler=None,
        target_index: int = 0,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare LSTM and naive persistence baseline at each
        forecast horizon (Day 1 ... Day N).
        """

        self._load_checkpoint()

        _, lstm_preds, targets = self._run_epoch(
            test_dl,
            train=False
        )

        # Naive baseline: last observed price repeated across horizon
        naive_preds = []

        for X_batch, y_batch in test_dl:
            last_price_scaled = X_batch[:, -1, target_index]
            horizon = y_batch.shape[1]

            baseline = last_price_scaled.unsqueeze(1).repeat(
                1, horizon
            )

            naive_preds.append(baseline.cpu().numpy())

        naive_preds = np.concatenate(naive_preds)

        # Convert scaled values back to ₹
        if target_scaler is not None:
            original_shape = lstm_preds.shape

            lstm_preds = target_scaler.inverse_transform(
                lstm_preds.reshape(-1, 1)
            ).reshape(original_shape)

            targets = target_scaler.inverse_transform(
                targets.reshape(-1, 1)
            ).reshape(original_shape)

            naive_preds = target_scaler.inverse_transform(
                naive_preds.reshape(-1, 1)
            ).reshape(original_shape)

        results = {}

        for h in range(targets.shape[1]):
            day = h + 1

            lstm_metrics = compute_metrics(
                targets[:, h],
                lstm_preds[:, h]
            )

            naive_metrics = compute_metrics(
                targets[:, h],
                naive_preds[:, h]
            )

            results[f"day_{day}"] = {
                "lstm_mae": lstm_metrics["mae"],
                "naive_mae": naive_metrics["mae"],
                "lstm_rmse": lstm_metrics["rmse"],
                "naive_rmse": naive_metrics["rmse"],
                "lstm_mape": lstm_metrics["mape"],
                "naive_mape": naive_metrics["mape"],
            }

        logger.info("Horizon-wise evaluation:")
        logger.info(
            "Day | LSTM MAE | Naive MAE | "
            "LSTM RMSE | Naive RMSE | LSTM MAPE | Naive MAPE"
        )

        for day, metrics in results.items():
            day_num = day.replace("day_", "")

            logger.info(
                f"{day_num:>3} | "
                f"{metrics['lstm_mae']:>9.2f} | "
                f"{metrics['naive_mae']:>9.2f} | "
                f"{metrics['lstm_rmse']:>10.2f} | "
                f"{metrics['naive_rmse']:>10.2f} | "
                f"{metrics['lstm_mape']:>9.2f}% | "
                f"{metrics['naive_mape']:>10.2f}%"
            )

        return results

    def evaluate_seasonal_naive_horizons(
        self,
        test_dl: DataLoader,
        target_scaler=None,
        target_index: int = 0,
        season_length: int = 7,
    ) -> Dict[int, Dict[str, float]]:
        """
        Evaluate seasonal-naive performance separately for
        every forecast horizon.
        """

        if season_length > self.cfg.sequence_length:
            raise ValueError(
                f"season_length ({season_length}) cannot exceed "
                f"sequence_length ({self.cfg.sequence_length})"
            )

        all_preds = []
        all_targets = []

        for X_batch, y_batch in test_dl:

            seasonal_history = X_batch[
                :, -season_length:, target_index
            ]

            horizon = y_batch.shape[1]

            seasonal_preds = torch.stack(
                [
                    seasonal_history[:, h % season_length]
                    for h in range(horizon)
                ],
                dim=1,
            )

            all_preds.append(seasonal_preds.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())

        preds = np.concatenate(all_preds, axis=0)
        targets = np.concatenate(all_targets, axis=0)

        # Inverse transform.
        if target_scaler is not None:
            preds = target_scaler.inverse_transform(
                preds.reshape(-1, 1)
            ).reshape(preds.shape)

            targets = target_scaler.inverse_transform(
                targets.reshape(-1, 1)
            ).reshape(targets.shape)

        results = {}

        for h in range(preds.shape[1]):
            metrics = compute_metrics(
                targets[:, h],
                preds[:, h],
            )

            results[h + 1] = metrics

            logger.info(
                f"Seasonal Naive Day {h + 1:02d} | "
                f"MAE: ₹{metrics['mae']:.2f} | "
                f"RMSE: ₹{metrics['rmse']:.2f} | "
                f"MAPE: {metrics['mape']:.2f}%"
            )

        return results

    # ── Checkpoint helpers ─────────────────────
    def _save_checkpoint(self, epoch: int, val_loss: float, metrics: dict):
        torch.save({
            "epoch":      epoch,
            "val_loss":   val_loss,
            "metrics":    metrics,
            "model_state": self.model.state_dict(),
            "optim_state": self.optimizer.state_dict(),
        }, self.ckpt_path)
        logger.debug(f"Checkpoint saved → {self.ckpt_path}")

    def _load_checkpoint(self):
        if os.path.exists(self.ckpt_path):
            ckpt = torch.load(self.ckpt_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state"])
            logger.info(f"Loaded best checkpoint from epoch {ckpt['epoch']}")
        else:
            logger.warning("No checkpoint found — using current model weights")
