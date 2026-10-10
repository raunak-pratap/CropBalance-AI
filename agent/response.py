from typing import Dict


def build_response(
    verified_results: Dict,
    status: str = "success",
    errors: Dict | None = None,
) -> Dict:
    """
    Convert verified CropBalance tool results into a
    human-readable response.

    This layer formats results only.
    It does not generate new agricultural facts.
    """

    sections = []
    errors = errors or {}

    # ---------------------------------------------------------
    # PRICE PREDICTION
    # ---------------------------------------------------------
    if "predict_price" in verified_results:
        data = verified_results["predict_price"]["result"]

        forecast = data.get("forecast", [])
        crop = data.get("crop", "crop")
        state = data.get("state", "your state")

        data_source = data.get("data_source", "unknown")
        is_live = data.get("is_live", False)

        source_note = ""

        if not is_live:
            source_note = (
                f" ⚠️ Data source: {data_source}. "
                "This is not live market data."
            )

        sections.append(
            f"📈 Price forecast for {crop} in {state}: "
            f"{len(forecast)} forecast points are available."
            f"{source_note}"
        )

    # ---------------------------------------------------------
    # WEATHER
    # ---------------------------------------------------------
    if "get_weather" in verified_results:
        data = verified_results["get_weather"]["result"]

        state = data.get("state", "your state")
        days = data.get("days", len(data.get("weather", [])))

        data_source = data.get("data_source", "unknown")
        is_live = data.get("is_live", False)

        source_note = ""

        if not is_live:
            source_note = (
                f" ⚠️ Data source: {data_source}. "
                "This is not live weather data."
            )

        sections.append(
            f"🌦️ Weather information for {state}: "
            f"{days} days of weather data are available."
            f"{source_note}"
        )

    # ---------------------------------------------------------
    # DISEASE DETECTION
    # ---------------------------------------------------------
    if "detect_disease" in verified_results:
        data = verified_results["detect_disease"]["result"]

        disease = data.get("disease", "Unknown disease")
        confidence = data.get("confidence")

        if confidence is not None:
            confidence_text = f"{confidence * 100:.1f}% confidence"
        else:
            confidence_text = "confidence unavailable"

        sections.append(
            f"🌿 Disease detection: {disease} "
            f"({confidence_text})."
        )

    # ---------------------------------------------------------
    # AGRICULTURE ADVICE
    # ---------------------------------------------------------
    if "get_agriculture_advice" in verified_results:
        data = verified_results["get_agriculture_advice"]["result"]

        severity = data.get("severity", "unknown")
        treatment_en = data.get("treatment_en", "")
        treatment_hi = data.get("treatment_hi", "")

        sections.append(
            f"🩺 Agriculture advice ({severity} severity): "
            f"{treatment_en}\n"
            f"🇮🇳 Hindi: {treatment_hi}\n"
            f"⚠️ {data.get('disclaimer', '')}"
        )

    # ---------------------------------------------------------
    # FARMER CONTEXT
    # ---------------------------------------------------------
    if "get_farmer_context" in verified_results:
        data = verified_results["get_farmer_context"]["result"]

        farmer_id = data.get("farmer_id", "unknown")
        state = data.get("state", "not provided")
        district = data.get("district", "not provided")
        current_crop = data.get("current_crop", "not provided")
        land_size = data.get("land_size", "not provided")
        soil_type = data.get("soil_type", "not provided")

        sections.append(
            "🌾 Your farm details:\n"
            f"Farmer ID: {farmer_id}\n"
            f"State: {state}\n"
            f"District: {district}\n"
            f"Current crop: {current_crop}\n"
            f"Land size: {land_size} acres\n"
            f"Soil type: {soil_type}"
        )

    # ---------------------------------------------------------
    # ERRORS
    # ---------------------------------------------------------
    if errors:
        if status == "partial_success":
            sections.append(
                "ℹ️ Some information is available, but "
                "other parts of your request could not be completed."
            )
        for tool, error in errors.items():
            message = error.get(
                "message",
                "This tool could not complete the request.",
            )

            sections.append(
                f"⚠️ {tool}: {message}"
            )
    
    # ---------------------------------------------------------
    # UPDATE FARMER CONTEXT
    # ---------------------------------------------------------
    if "update_farmer_context" in verified_results:
        data = verified_results["update_farmer_context"]["result"]

        farmer_id = data.get("farmer_id", "unknown")
        current_crop = data.get("current_crop")
        state = data.get("state")
        district = data.get("district")

        updated_fields = []

        if current_crop:
            updated_fields.append(f"Current crop: {current_crop}")

        if state:
            updated_fields.append(f"State: {state}")

        if district:
            updated_fields.append(f"District: {district}")

        details = "\n".join(updated_fields)

        sections.append(
            "✅ Your farmer profile has been updated successfully.\n"
            f"Farmer ID: {farmer_id}\n"
            f"{details}"
        )

    # ---------------------------------------------------------
    # FALLBACK
    # ---------------------------------------------------------
    if not sections:
        sections.append(
            "I could not produce a result for this request."
        )


    return {
        "status": status,
        "message": "\n".join(sections),
    }