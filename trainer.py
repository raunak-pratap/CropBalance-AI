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

    def evaluate(self, test_dl: DataLoader) -> Dict[str, float]:
        """Evaluate on test set using best saved checkpoint."""
        self._load_checkpoint()
        _, preds, targets = self._run_epoch(test_dl, train=False)
        metrics = compute_metrics(targets.flatten(), preds.flatten())
        logger.info(f"Test metrics: {metrics}")
        return metrics

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
