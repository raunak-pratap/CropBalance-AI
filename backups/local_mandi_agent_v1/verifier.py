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

    if tool == "get_agriculture_advice":
        advice_status = result.get("status")
        treatment_en = result.get("treatment_en")
        treatment_hi = result.get("treatment_hi")

        issues = []

        if advice_status != "success":
            issues.append("Agriculture advice was not generated successfully.")

        if not treatment_en:
            issues.append("Agriculture advice contains no English treatment guidance.")

        if not treatment_hi:
            issues.append("Agriculture advice contains no Hindi treatment guidance.")

        if issues:
            return {
                "valid": False,
                "issues": issues,
            }

        return {
            "valid": True,
            "issues": [],
            "checks": {
                "advice_present": True,
                "severity": result.get("severity", "unknown"),
                "source": result.get("source", "unknown"),
            },
        }

    
    elif tool == "update_farmer_context":
        issues = []

        if not isinstance(result, dict):
            issues.append("Update result must be a dictionary.")
        elif result.get("status") != "success":
            issues.append("Farmer context update was not successful.")
        elif not result.get("farmer_id"):
            issues.append("Missing farmer ID in update result.")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
        }


    elif tool == "get_farmer_context":
        context_status = result.get("status")
        farmer_id = result.get("farmer_id")
        state = result.get("state")
        district = result.get("district")

        issues = []

        if context_status != "success":
            issues.append("Farmer context was not retrieved successfully.")

        if not farmer_id:
            issues.append("Farmer context contains no farmer ID.")

        if not state:
            issues.append("Farmer context contains no state information.")

        if not district:
            issues.append("Farmer context contains no district information.")

        if issues:
            return {
                "valid": False,
                "issues": issues,
            }

        return {
            "valid": True,
            "issues": [],
            "checks": {
                "context_present": True,
                "farmer_id": farmer_id,
                "state": state,
                "district": district,
            },
        }

    return {
        "valid": False,
        "issues": [f"No verifier exists for tool: {tool}"],
    }