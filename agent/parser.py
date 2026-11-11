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
    "Chhattisgarh",
    "Chennai",
    "Delhi",
    "Goa",
    "Telangana",
    "Jharkhand",
    "Jammu and Kashmir",
    "Nagaland",
    "Assam",
    "Arunachal Pradesh",
    "Manipur",
    "Meghalaya",
    "Mizoram",
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

farm_context_keywords = [
    "my farm",
    "my farm details",
    "my farm information",
    "my land",
    "my soil",
    "my soil type",
    "my current crop",
    "current crop",
    "my location",
    "my state",
    "my district",
    "farm details",
    "farmer details",
    "i grow",
]

farmer_context_update_keywords = [
    "i live in",
    "i am from",
    "my soil is",
    "my crop is",
    "my farm is",
    "my land is",
    "my state is",
    "i grow",
    "update my crop",
    "change my crop",
    "update my state",
    "change my state",
    "update my soil",
    "change my soil",
    "update my land",
    "change my land",
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

def is_local_mandi_request(text: str) -> bool:
    """Identify requests for archived local mandi information."""

    text = text.casefold()

    # Explicit forecasts should continue using the LSTM price tool.
    forecast_terms = [
        "forecast",
        "predict",
        "prediction",
        "future price",
        "future prices",
        "expected price",
        "expected prices",
    ]

    if any(term in text for term in forecast_terms):
        return False

    local_markers = [
        "mandi",
        "apmc",
        "local market",
        "local markets",
        "local price",
        "local prices",
        "nearby market",
        "nearby markets",
        "market price",
        "market prices",
        "compare market",
        "compare markets",
        "market-wise",
        "market wise",
        "prices in dharashiv",
        "price in dharashiv",
        "dharashiv prices",
    ]

    if any(marker in text for marker in local_markers):
        return True

    # Osmanabad is an older name for the district represented
    # by the current Dharashiv archive.
    has_dharashiv_location = (
        "dharashiv" in text or "osmanabad" in text
    )

    has_price_context = any(
        word in text.split()
        for word in ["price", "prices", "market", "bhav"]
    )

    return has_dharashiv_location and has_price_context


def is_market_comparison_request(text: str) -> bool:
    """Determine whether the user wants markets ranked or compared."""

    text = text.casefold()

    comparison_terms = [
        "compare",
        "comparison",
        "highest",
        "rank",
        "ranked",
        "which market",
        "which mandi",
        "top market",
        "top mandi",
        "best market",
        "best mandi",
    ]

    return any(term in text for term in comparison_terms)


def extract_district(text: str) -> Optional[str]:
    """Recognise the district currently supported by the local archive."""

    text = text.casefold()

    if "dharashiv" in text or "osmanabad" in text:
        return "Dharashiv"

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

    if is_local_mandi_request(text):
        return "local_mandi"

    elif any(keyword in text for keyword in price_keywords):
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

    if any(
        keyword in text
        for keyword in farmer_context_update_keywords
    ):
        return "farmer_context_update"

    if any(keyword in text for keyword in farm_context_keywords):
        return "farm_context"

    return "unknown"

def extract_intents(text: str) -> list[str]:
    """Extract all supported intents from the user's request."""

    text = text.lower()

    intents = []

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

    weather_keywords = [
        "weather",
        "temperature",
        "rain",
        "rainfall",
        "humidity",
        "climate",
    ]

    advice_keywords = [
        "what should i do",
        "what can i do",
        "how should i treat",
        "how can i treat",
        "treatment",
        "treat",
        "solution",
        "remedy",
        "advice",
        "recommend",
        "recommendation",
        "medicine",
        "pesticide",
    ]

    if any(keyword in text for keyword in disease_keywords):
        intents.append("disease_detection")

    if is_local_mandi_request(text):
        intents.append("local_mandi")

    elif any(keyword in text for keyword in price_keywords):
        intents.append("price_prediction")

    if any(keyword in text for keyword in weather_keywords):
        intents.append("weather")

    if any(keyword in text for keyword in advice_keywords):
        intents.append("agriculture_advice")

    if any(
        keyword in text
        for keyword in farmer_context_update_keywords
    ):
        intents.append("farmer_context_update")

    if any(keyword in text for keyword in farm_context_keywords):
        intents.append("farm_context")

    if not intents:
        intents.append("unknown")
    
    return intents


def parse_request(text: str) -> Dict:
    """Extract intents and basic entities from a CropBalance user request."""

    text = text.strip()

    return {
        "intent": extract_intent(text),
        "intents": extract_intents(text),
        "crop": extract_crop(text),
        "state": extract_state(text),
        "district": extract_district(text),
        "compare_markets": is_market_comparison_request(text),
    }