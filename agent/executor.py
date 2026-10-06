from typing import Dict

from agent.tools import (
    detect_disease,
    predict_price,
    get_weather,
    get_agriculture_advice,
)


def _execute_tool(
    tool: str,
    parsed: Dict,
    image_path: str | None = None,
    crop: str | None = None,
    state: str | None = None,
    disease_result: Dict | None = None,
) -> Dict:
    """Execute one CropBalance tool."""

    if tool == "detect_disease":
        if not image_path:
            return {
                "status": "missing_input",
                "missing": ["image"],
                "message": "Please upload a clear image of the crop leaf.",
                "result": None,
            }

        result = detect_disease(image_path)

        return {
            "status": "success",
            "result": result,
        }

    if tool == "predict_price":
        crop = crop or parsed.get("crop")
        state = state or parsed.get("state")

        missing = []

        if not crop:
            missing.append("crop")

        if not state:
            missing.append("state")

        if missing:
            return {
                "status": "missing_input",
                "missing": missing,
                "message": f"Please provide your {', '.join(missing)}.",
                "result": None,
            }

        result = predict_price(
            crop=crop,
            state=state,
        )

        return {
            "status": "success",
            "result": result,
        }

    if tool == "get_weather":
        state = state or parsed.get("state")

        if not state:
            return {
                "status": "missing_input",
                "missing": ["state"],
                "message": "Please provide your state.",
                "result": None,
            }

        result = get_weather(
            state=state,
        )

        return {
            "status": "success",
            "result": result,
        }


    if tool == "get_agriculture_advice":
        # Advice must be grounded in an actual disease result.
        # Never invent disease-specific treatment when image analysis
        # has not completed successfully.
        disease_result = disease_result or {}
        disease = disease_result.get("disease")
        confidence = disease_result.get("confidence")

        if not disease:
            return {
                "status": "missing_input",
                "missing": ["disease_detection"],
                "message": (
                    "Disease-specific advice requires a successful "
                    "disease detection first. Please upload a clear "
                    "image of the affected crop leaf."
                ),
                "result": None,
            }

        result = get_agriculture_advice(
            disease=disease,
            confidence=confidence,
        )

        if result.get("status") != "success":
            return {
                "status": "error",
                "message": result.get(
                    "message",
                    "Agriculture advice could not be generated.",
                ),
                "result": None,
            }

        return {
            "status": "success",
            "result": result,
        }

    return {
        "status": "unsupported",
        "message": f"Unknown tool: {tool}",
        "result": None,
    }


def execute_plan(
    plan: Dict,
    parsed: Dict,
    image_path: str | None = None,
    crop: str | None = None,
    state: str | None = None,
) -> Dict:
    """
    Execute one or multiple tools selected by the planner.

    Successful tool results are preserved even if another
    tool fails.
    """

    tools = plan.get("tools")

    # Backward compatibility with old single-tool plans
    if not tools:
        tool = plan.get("tool")

        if tool:
            tools = [tool]
        else:
            return {
                "status": "unsupported",
                "message": "No executable tool found in the plan.",
                "result": {},
                "errors": {},
            }

    results = {}
    errors = {}
    disease_result = None

    for tool in tools:
        try:
            execution = _execute_tool(
                tool=tool,
                parsed=parsed,
                image_path=image_path,
                crop=crop,
                state=state,
                disease_result=disease_result,
            )

            if execution["status"] == "success":
                results[tool] = execution["result"]

                if tool == "detect_disease":
                    disease_result = execution["result"]
            else:
                errors[tool] = {
                    "status": execution["status"],
                    "missing": execution.get("missing", []),
                    "message": execution.get("message"),
                }

        except Exception as exc:
            errors[tool] = {
                "status": "error",
                "message": str(exc),
            }

    # Nothing succeeded
    if not results and errors:
        return {
            "status": "error",
            "result": {},
            "errors": errors,
        }

    # Some tools succeeded, some failed
    if results and errors:
        return {
            "status": "partial_success",
            "result": results,
            "errors": errors,
        }

    # Everything succeeded
    return {
        "status": "success",
        "result": results,
        "errors": {},
    }