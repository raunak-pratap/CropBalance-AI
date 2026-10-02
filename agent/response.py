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

    if "predict_price" in verified_results:
        data = verified_results["predict_price"]["result"]

        forecast = data.get("forecast", [])
        crop = data.get("crop", "crop")
        state = data.get("state", "your state")

        sections.append(
            f"📈 Price forecast for {crop} in {state}: "
            f"{len(forecast)} forecast points are available."
        )

    if "get_weather" in verified_results:
        data = verified_results["get_weather"]["result"]

        state = data.get("state", "your state")
        days = data.get("days", len(data.get("weather", [])))

        sections.append(
            f"🌦️ Weather information for {state}: "
            f"{days} days of weather data are available."
        )

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

    if errors:
        for tool, error in errors.items():
            message = error.get(
                "message",
                "This tool could not complete the request.",
            )

            sections.append(
                f"⚠️ {tool}: {message}"
            )

    if not sections:
        sections.append(
            "I could not produce a result for this request."
        )

    return {
        "status": status,
        "message": "\n".join(sections),
    }