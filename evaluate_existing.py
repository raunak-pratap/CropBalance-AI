import os
from datetime import datetime

from loguru import logger

from fetcher import MandiPriceFetcher, WeatherFetcher
from preprocessor import run_preprocessing_pipeline
from lstm_model import build_model
from trainer import Trainer


CROP = "tomato"
STATE = "Maharashtra"
START_DATE = "2024-10-01"
END_DATE = "2026-09-30"


def main():

    logger.info("=" * 60)
    logger.info("Existing Model Evaluation")
    logger.info("=" * 60)

    # 1. Fetch the exact same data period
    price_fetcher = MandiPriceFetcher()
    weather_fetcher = WeatherFetcher()

    price_df = price_fetcher.fetch(
        CROP,
        STATE,
        START_DATE,
        END_DATE,
    )

    weather_df = weather_fetcher.fetch(
        STATE,
        START_DATE,
        END_DATE,
    )

    # 2. Recreate preprocessing/test set
    train_dl, val_dl, test_dl, meta = run_preprocessing_pipeline(
        price_df,
        weather_df,
        crop=CROP,
        save=False,
    )

    target_scaler = meta["scalers"]["modal_price"]
    target_index = meta["feature_columns"].index("modal_price")

    # 3. Build model architecture
    model = build_model(
        n_features=meta["n_features"]
    )

    trainer = Trainer(
        model=model,
        crop=CROP,
    )

    logger.info("=" * 60)
    logger.info("PERSISTENCE BASELINE")
    logger.info("=" * 60)

    persistence = trainer.evaluate_naive_baseline(
        test_dl,
        target_scaler=target_scaler,
        target_index=target_index,
    )

    logger.info(
        f"Persistence → "
        f"MAE={persistence['mae']:.2f}, "
        f"RMSE={persistence['rmse']:.2f}, "
        f"MAPE={persistence['mape']:.2f}%"
    )

    logger.info("=" * 60)
    logger.info("7-DAY SEASONAL NAIVE")
    logger.info("=" * 60)

    seasonal = trainer.evaluate_seasonal_naive_baseline(
        test_dl,
        target_scaler=target_scaler,
        target_index=target_index,
        season_length=7,
    )

    logger.info(
        f"Seasonal-naive → "
        f"MAE={seasonal['mae']:.2f}, "
        f"RMSE={seasonal['rmse']:.2f}, "
        f"MAPE={seasonal['mape']:.2f}%"
    )

    logger.info("=" * 60)
    logger.info("LSTM")
    logger.info("=" * 60)

    lstm = trainer.evaluate(
        test_dl,
        target_scaler=target_scaler,
    )

    logger.info(
        f"LSTM → "
        f"MAE={lstm['mae']:.2f}, "
        f"RMSE={lstm['rmse']:.2f}, "
        f"MAPE={lstm['mape']:.2f}%"
    )

    logger.info("=" * 60)
    logger.info("FINAL COMPARISON")
    logger.info("=" * 60)

    logger.info(
        f"{'Model':<20} "
        f"{'MAE':>10} "
        f"{'RMSE':>10} "
        f"{'MAPE':>10}"
    )

    logger.info(
        f"{'Persistence':<20} "
        f"{persistence['mae']:>10.2f} "
        f"{persistence['rmse']:>10.2f} "
        f"{persistence['mape']:>9.2f}%"
    )

    logger.info(
        f"{'Seasonal-naive 7d':<20} "
        f"{seasonal['mae']:>10.2f} "
        f"{seasonal['rmse']:>10.2f} "
        f"{seasonal['mape']:>9.2f}%"
    )

    logger.info(
        f"{'LSTM':<20} "
        f"{lstm['mae']:>10.2f} "
        f"{lstm['rmse']:>10.2f} "
        f"{lstm['mape']:>9.2f}%"
    )


if __name__ == "__main__":
    main()