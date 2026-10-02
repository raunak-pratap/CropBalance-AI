from typing import Dict

from agent.tools import (
    detect_disease,
    predict_price,
    get_weather,
)


def execute_plan(
    plan: Dict,
    parsed: Dict,
    image_path: str | None = None,
    crop: str | None = None,
    state: str | None = None,
) -> Dict:
    """
    Execute the tool selected by the planner.

    The executor is responsible for:
        plan → validate inputs → tool → result
    """

    tool = plan.get("tool")

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

    return {
        "status": "unsupported",
        "message": f"Unknown tool: {tool}",
        "result": None,
    }