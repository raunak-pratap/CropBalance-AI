from typing import Dict


def verify_result(tool: str, result: Dict) -> Dict:
    """
    Perform basic structural verification of a tool result.

    This does not verify scientific correctness.
    It verifies that the expected result structure exists.
    """

    if not isinstance(result, dict):
        return {
            "valid": False,
            "issues": ["Tool result is not a dictionary."],
        }

    if tool == "predict_price":
        forecast = result.get("forecast")

        if not forecast:
            return {
                "valid": False,
                "issues": ["Price prediction contains no forecast data."]
            }

        data_source = result.get("data_source", "unknown")
        is_live = result.get("is_live", False)

        return {
            "valid": True,
            "issues": [],
            "checks": {
                "forecast_present": True,
                "forecast_points": len(forecast),
                "data_source": data_source,
                "is_live": is_live,
            }
        }

    if tool == "get_weather":
        weather = result.get("weather")

        if not weather:
            return {
                "valid": False,
                "issues": ["Weather result contains no weather records."]
            }

        data_source = result.get("data_source", "unknown")
        is_live = result.get("is_live", False)

        return {
            "valid": True,
            "issues": [],
            "checks": {
                "weather_present": True,
                "weather_records": len(weather),
                "data_source": data_source,
                "is_live": is_live,
            }
        }

    if tool == "detect_disease":
        disease = result.get("disease")
        confidence = result.get("confidence")

        issues = []

        if not disease:
            issues.append("Disease result contains no disease label.")

        if confidence is None:
            issues.append("Disease result contains no confidence score.")

        if issues:
            return {
                "valid": False,
                "issues": issues,
            }

        return {
            "valid": True,
            "issues": [],
            "checks": {
                "disease_present": True,
                "confidence_present": True,
            },
        }

    return {
        "valid": False,
        "issues": [f"No verifier exists for tool: {tool}"],
    }