from typing import Dict

from services.disease_service import DiseaseService
from services.price_service import PriceService


# Load services once per application process.
disease_service = DiseaseService()
price_service = PriceService()


def detect_disease(image_path: str) -> Dict:
    """
    Detect a crop disease from an image.
    """
    return disease_service.detect_from_path(image_path)


def predict_price(
    crop: str,
    state: str,
    days_history: int = 90,
) -> Dict:
    """
    Predict future crop prices using the trained LSTM.
    """
    return price_service.predict_price(
        crop=crop,
        state=state,
        days_history=days_history,
    )