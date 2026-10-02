from datetime import datetime, timedelta
from typing import Dict

from fetcher import WeatherFetcher


class WeatherService:
    """
    Service layer for CropBalance weather data.

    Uses the existing WeatherFetcher and keeps weather
    infrastructure separate from the agent layer.
    """

    def __init__(self):
        self.fetcher = WeatherFetcher()

    def get_weather(
        self,
        state: str,
        days_history: int = 7,
    ) -> Dict:
        if not state:
            raise ValueError("State is required for weather data.")

        if days_history < 1:
            raise ValueError("days_history must be at least 1.")

        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days_history - 1)

        df = self.fetcher.fetch(
            state=state,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
        )

        return {
            "state": state,
            "days": len(df),
            "weather": df.to_dict(orient="records"),
            "generated_at": datetime.now().isoformat(),
        }