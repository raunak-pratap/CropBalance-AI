from typing import Dict, Optional


CROPS = [
    "wheat",
    "rice",
    "tomato",
    "onion",
    "potato",
    "cotton",
    "soybean",
    "maize",
    "barley",
    "sugarcane",
]


STATES = [
    "Andhra Pradesh",
    "Bihar",
    "Gujarat",
    "Haryana",
    "Karnataka",
    "Madhya Pradesh",
    "Maharashtra",
    "Odisha",
    "Punjab",
    "Rajasthan",
    "Uttar Pradesh",
]


def extract_crop(text: str) -> Optional[str]:
    """Extract a supported crop name from the user's request."""

    import re

    text = text.lower()

    for crop in CROPS:
        pattern = rf"\b{re.escape(crop)}\b"

        if re.search(pattern, text):
            return crop

    return None


def extract_state(text: str) -> Optional[str]:
    """Extract a supported Indian state from the user's request."""

    text_lower = text.lower()

    for state in STATES:
        if state.lower() in text_lower:
            return state

    return None


def extract_intent(text: str) -> str:
    """Determine the user's main CropBalance intent."""

    text = text.lower()

    disease_keywords = [
        "disease",
        "diseased",
        "infection",
        "infected",
        "leaf",
        "leaves",
    ]

    price_keywords = [
        "price",
        "prices",
        "forecast",
        "predict",
        "prediction",
        "market",
        "sell",
        "selling",
        "bhav",
    ]

    if any(keyword in text for keyword in disease_keywords):
        return "disease_detection"

    if any(keyword in text for keyword in price_keywords):
        return "price_prediction"

    weather_keywords = [
    "weather",
    "temperature",
    "rain",
    "rainfall",
    "humidity",
    "climate",
]

    if any(keyword in text for keyword in weather_keywords):
        return "weather"

    return "unknown"


def parse_request(text: str) -> Dict:
    """Extract intent and basic entities from a CropBalance user request."""

    text = text.strip()

    return {
        "intent": extract_intent(text),
        "crop": extract_crop(text),
        "state": extract_state(text),
    }