from typing import Dict


def plan_request(parsed: Dict) -> Dict:
    """
    Create a deterministic execution plan from parsed user input.

    The parser is responsible for understanding the request.
    The planner is responsible for selecting the appropriate tool.
    """

    intent = parsed.get("intent")

    if intent == "disease_detection":
        return {
            "intent": intent,
            "tool": "detect_disease",
            "reason": "The parsed request requires crop disease detection.",
        }

    if intent == "price_prediction":
        return {
            "intent": intent,
            "tool": "predict_price",
            "reason": "The parsed request requires crop price prediction.",
        }

    return {
        "intent": "unknown",
        "tool": None,
        "reason": "No supported CropBalance capability matched the parsed request.",
    }