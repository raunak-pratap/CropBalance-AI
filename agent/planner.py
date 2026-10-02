from typing import Dict


INTENT_TO_TOOL = {
    "disease_detection": "detect_disease",
    "price_prediction": "predict_price",
    "weather": "get_weather",
}


def plan_request(parsed: Dict) -> Dict:
    """
    Create a deterministic execution plan from parsed user input.

    The parser understands the request.
    The planner maps intents to tools.
    """

    intents = parsed.get("intents", [])

    # Backward compatibility with older parsed requests
    if not intents:
        intent = parsed.get("intent")

        if intent:
            intents = [intent]

    # Remove unsupported intents
    tools = [
        INTENT_TO_TOOL[intent]
        for intent in intents
        if intent in INTENT_TO_TOOL
    ]

    # No supported intent
    if not tools:
        return {
            "intent": "unknown",
            "tools": [],
            "reason": (
                "No supported CropBalance capability "
                "matched the parsed request."
            ),
        }

    # Single-tool request
    if len(tools) == 1:
        return {
            "intent": intents[0],
            "tool": tools[0],
            "tools": tools,
            "reason": (
                f"The parsed request requires the "
                f"{tools[0]} tool."
            ),
        }

    # Multi-tool request
    return {
        "intent": "multi_tool",
        "tools": tools,
        "reason": (
            "The parsed request requires multiple "
            "CropBalance tools."
        ),
    }