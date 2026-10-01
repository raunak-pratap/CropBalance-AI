from typing import Dict

from services.disease_service import DiseaseService
from services.price_service import PriceService


_disease_service = None
_price_service = None


def _get_disease_service() -> DiseaseService:
    global _disease_service

    if _disease_service is None:
        _disease_service = DiseaseService()

    return _disease_service


def _get_price_service() -> PriceService:
    global _price_service

    if _price_service is None:
        _price_service = PriceService()

    return _price_service


def detect_disease(image_path: str) -> Dict:
    """
    Detect a crop disease from an image.

    The disease service is initialized only when this tool is used.
    """
    service = _get_disease_service()

    return service.detect_from_path(image_path)


def predict_price(
    crop: str,
    state: str,
    days_history: int = 90,
) -> Dict:
    """
    Predict future crop prices using the trained LSTM.

    The price service is initialized only when this tool is used.
    """
    service = _get_price_service()

    return service.predict_price(
        crop=crop,
        state=state,
        days_history=days_history,
    )