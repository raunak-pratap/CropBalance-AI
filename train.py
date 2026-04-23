"""
train.py
--------
End-to-end training script.

Usage:
    python train.py --crop wheat --state Punjab --years 5
    python train.py --crop tomato --state Maharashtra --years 3

The script will:
  1. Fetch/generate price + weather data
  2. Preprocess and build sequences
  3. Train LSTM with early stopping
  4. Evaluate on held-out test set
  5. Save model, scalers, and training history
"""

import argparse
import os
import sys
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend for servers
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from loguru import logger

from config import LSTM_CONFIG, PATH_CONFIG, SUPPORTED_CROPS, API_CONFIG
from fetcher import MandiPriceFetcher, WeatherFetcher
from preprocessor import run_preprocessing_pipeline
from lstm_model import build_model
from trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train Crop Price LSTM")
    parser.add_argument("--crop",  type=str, default="wheat",
                        choices=SUPPORTED_CROPS, help="Crop to train")
    parser.add_argument("--state", type=str, default="Punjab",
                        help="Indian state for mandi data")
    parser.add_argument("--years", type=int, default=5,
                        help="Years of historical data to use")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override config epochs")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip saving training curves")
    return parser.parse_args()


def plot_training_curves(history: dict, crop: str):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Training Curves — {crop.title()}", fontsize=14, fontweight="bold")

    axes[0, 0].plot(history["train_loss"], label="Train loss", color="#2563EB")
    axes[0, 0].plot(history["val_loss"],   label="Val loss",   color="#DC2626")
    axes[0, 0].set_title("Loss (Huber)")
    axes[0, 0].legend()

    axes[0, 1].plot(history["val_mae"], color="#059669")
    axes[0, 1].set_title("Validation MAE (scaled)")

    axes[1, 0].plot(history["val_rmse"], color="#D97706")
    axes[1, 0].set_title("Validation RMSE (scaled)")

    axes[1, 1].plot(history["val_mape"], color="#7C3AED")
    axes[1, 1].set_title("Validation MAPE (%)")

    for ax in axes.flatten():
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(PATH_CONFIG.models_dir, f"training_curves_{crop}.png")
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info(f"Training curves saved → {plot_path}")


def main():
    args = parse_args()

    # ── Setup ──────────────────────────────────
    PATH_CONFIG.create_dirs()
    logger.add(
        os.path.join(PATH_CONFIG.logs_dir, f"train_{args.crop}_{datetime.now():%Y%m%d_%H%M%S}.log"),
        level="DEBUG",
    )

    if args.epochs:
        LSTM_CONFIG.epochs = args.epochs

    end_date   = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=args.years * 365)).strftime("%Y-%m-%d")

    logger.info("=" * 60)
    logger.info(f"Smart Farming — Crop Price Prediction")
    logger.info(f"Crop: {args.crop} | State: {args.state}")
    logger.info(f"Period: {start_date} → {end_date}")
    logger.info("=" * 60)

    # ── 1. Fetch data ──────────────────────────
    logger.info("Step 1/4: Fetching data...")
    price_fetcher   = MandiPriceFetcher()
    weather_fetcher = WeatherFetcher()

    price_df   = price_fetcher.fetch(args.crop, args.state, start_date, end_date)
    weather_df = weather_fetcher.fetch(args.state, start_date, end_date)

    logger.info(f"Price rows: {len(price_df)} | Weather rows: {len(weather_df)}")

    # ── 2. Preprocess ──────────────────────────
    logger.info("Step 2/4: Preprocessing...")
    train_dl, val_dl, test_dl, meta = run_preprocessing_pipeline(
        price_df, weather_df, crop=args.crop
    )
    n_features = meta["n_features"]

    # ── 3. Train ───────────────────────────────
    logger.info("Step 3/4: Training LSTM...")
    model   = build_model(n_features=n_features)
    trainer = Trainer(model=model, crop=args.crop)
    history = trainer.train(train_dl, val_dl)

    # ── 4. Evaluate ────────────────────────────
    logger.info("Step 4/4: Evaluating on test set...")
    test_metrics = trainer.evaluate(test_dl)
    logger.info(f"Final test metrics:")
    logger.info(f"  MAE:  {test_metrics['mae']:.4f}")
    logger.info(f"  RMSE: {test_metrics['rmse']:.4f}")
    logger.info(f"  MAPE: {test_metrics['mape']:.2f}%")

    # ── Save plots ─────────────────────────────
    if not args.no_plot:
        plot_training_curves(history, args.crop)

    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info(f"Model saved → {PATH_CONFIG.models_dir}/best_{args.crop}.pt")
    logger.info(f"Scalers saved → {PATH_CONFIG.scalers_dir}/scalers_{args.crop}.pkl")
    logger.info("=" * 60)
    logger.info(f"Run the API: uvicorn api.main:app --reload")


if __name__ == "__main__":
    main()
