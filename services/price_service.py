from datetime import datetime, timedelta
from typing import Dict

import pandas as pd

from fetcher import MandiPriceFetcher, WeatherFetcher
from predictor import CropPredictor


class PriceService:
    """
    Service layer for crop price forecasting.

    Combines historical mandi prices and weather data,
    then sends the resulting dataset to the trained LSTM.
    """

    def __init__(self):
        self.mandi_fetcher = MandiPriceFetcher()
        self.weather_fetcher = WeatherFetcher()

        # Cache predictors so each crop model is loaded only once.
        self.predictors = {}

    def _get_predictor(self, crop: str) -> CropPredictor:
        crop = crop.lower()

        if crop not in self.predictors:
            self.predictors[crop] = CropPredictor(crop=crop)

        return self.predictors[crop]

    def predict_price(
        self,
        crop: str,
        state: str,
        days_history: int = 120,
    ) -> Dict:
        """
        Generate a future crop-price forecast.

        Parameters
        ----------
        crop:
            Crop name, e.g. "wheat".

        state:
            Indian state, e.g. "Punjab".

        days_history:
            Number of historical days used by the model.
        """

        if days_history < 60:
            raise ValueError(
                "days_history must be at least 60 because "
                "the LSTM sequence length is 60."
            )

        crop = crop.lower()

        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days_history - 1)

        start = start_date.isoformat()
        end = end_date.isoformat()

        # Fetch historical mandi prices.
        price_df = self.mandi_fetcher.fetch(
            crop=crop,
            state=state,
            start_date=start,
            end_date=end,
        )

        # Fetch historical weather.
        weather_df = self.weather_fetcher.fetch(
            state=state,
            start_date=start,
            end_date=end,
        )

        # Make sure both date columns are datetime.
        price_df["date"] = pd.to_datetime(price_df["date"])
        weather_df["date"] = pd.to_datetime(weather_df["date"])

        # Combine market + weather information.
        df = price_df.merge(
            weather_df,
            on=["date", "state"],
            how="left",
        )

        # Validate the features required by the trained model.
        required_features = [
            "modal_price",
            "min_price",
            "max_price",
            "arrivals_tonnes",
            "temp_max",
            "temp_min",
            "rainfall_mm",
            "humidity_pct",
        ]

        missing = [
            column
            for column in required_features
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required prediction features: {missing}"
            )

        # Run the trained LSTM.
        predictor = self._get_predictor(crop)

        result = predictor.predict(df)
        result["state"] = state
        result["history_days"] = days_history

        result["data_source"] = price_df.attrs.get(
            "data_source",
            "unknown",
        )
        result["is_live"] = price_df.attrs.get(
            "is_live",
            False,
        )

        return result