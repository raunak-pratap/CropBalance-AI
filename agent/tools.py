from typing import Dict

from services.disease_service import DiseaseService
from services.price_service import PriceService
from services.weather_service import WeatherService
from services.agriculture_advice_service import AgricultureAdviceService
from agent.context.postgres_memory import PostgresContextStore

_disease_service = None
_price_service = None
_weather_service = None
_agriculture_advice_service = None
_context_store = None


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
    days_history: int = 120,
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

def _get_weather_service() -> WeatherService:
    global _weather_service

    if _weather_service is None:
        _weather_service = WeatherService()

    return _weather_service


def get_weather(
    state: str,
    days_history: int = 7,
) -> Dict:
    """
    Get weather data for a state.
    """
    service = _get_weather_service()

    return service.get_weather(
        state=state,
        days_history=days_history,
    )

def _get_agriculture_advice_service() -> AgricultureAdviceService:
    global _agriculture_advice_service

    if _agriculture_advice_service is None:
        _agriculture_advice_service = AgricultureAdviceService()

    return _agriculture_advice_service


def get_agriculture_advice(
    disease: str | None = None,
    confidence: float | None = None,
) -> Dict:
    """
    Get disease-specific agriculture advice from CropBalance metadata.
    """
    service = _get_agriculture_advice_service()

    return service.get_advice(
        disease=disease,
        confidence=confidence,
    )

def _get_context_store() -> PostgresContextStore:
    global _context_store

    if _context_store is None:
        _context_store = PostgresContextStore()

    return _context_store

def update_farmer_context(
    farmer_id: str,
    crop: str | None = None,
    state: str | None = None,
    district: str | None = None,
    land_size: float | None = None,
    soil_type: str | None = None,
) -> Dict:
    """
    Update a farmer's stored context.

    Only values that are provided are updated.
    Existing values are preserved when an argument is None.
    """

    store = _get_context_store()

    context = store.get(farmer_id=farmer_id)

    if context is None:
        return {
            "status": "not_found",
            "farmer_id": farmer_id,
        }

    updated_context = store.update(
        farmer_id=farmer_id,
        current_crop=crop,
        state=state,
        district=district,
        land_size=land_size,
        soil_type=soil_type,
    )

    return {
        "status": "success",
        "farmer_id": updated_context.farmer_id,
        "state": updated_context.state,
        "district": updated_context.district,
        "current_crop": updated_context.current_crop,
        "land_size": updated_context.land_size,
        "soil_type": updated_context.soil_type,
    }