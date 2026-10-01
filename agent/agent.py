from typing import Dict

from agent.planner import plan_request
from agent.parser import parse_request
from agent.tools import detect_disease, predict_price



class CropBalanceAgent:
    """
    CropBalance Agent v0.1

    Simple deterministic agent:
        request → plan → tool → result
    """

    def __init__(self):
        self.pending_request = None

    def _complete_pending_request(self, request: str) -> Dict | None:
        """Try to complete a previously incomplete request."""

        if not self.pending_request:
            return None

        parsed = parse_request(request)

        pending = self.pending_request

        crop = pending.get("crop") or parsed.get("crop")
        state = pending.get("state") or parsed.get("state")

        if pending["intent"] == "price_prediction":
            if crop and state:
                self.pending_request = None

                return {
                    "intent": "price_prediction",
                    "crop": crop,
                    "state": state,
                }

        return None

    def run(
        self,
        request: str,
        image_path: str | None = None,
        crop: str | None = None,
        state: str | None = None,
    ) -> Dict:
                # 1. Try to complete a previous request first
        completed = self._complete_pending_request(request)

        if completed:
            parsed = completed
        else:
            # 2. Parse the new request
            parsed = parse_request(request)

        # 3. Create an execution plan
        plan = plan_request(parsed)

        # 2. Check whether we understand the request
        if plan["tool"] is None:
            return {
                "status": "unsupported",
                "plan": plan,
                "result": None,
            }

        # 3. Execute selected tool
        if plan["tool"] == "detect_disease":
            if not image_path:
                return {
                    "status": "missing_input",
                    "plan": plan,
                    "parsed": parsed,
                    "result": None,
                    "missing": ["image"],
                    "message": "Please upload a clear image of the crop leaf.",
                }

            result = detect_disease(image_path)

        elif plan["tool"] == "predict_price":
            crop = crop or parsed["crop"]
            state = state or parsed["state"]

            if not crop or not state:
                missing = []

                if not crop:
                    missing.append("crop")

                if not state:
                    missing.append("state")

                self.pending_request = {
                    "intent": parsed["intent"],
                    "crop": crop,
                    "state": state,
                    "missing": missing,
                }

                return {
                    "status": "missing_input",
                    "plan": plan,
                    "parsed": parsed,
                    "result": None,
                    "missing": missing,
                    "message": (
                        f"Please provide your {', '.join(missing)}."
                    ),
                }

            result = predict_price(
                crop=crop,
                state=state,
            )

        else:
            return {
                "status": "unsupported",
                "plan": plan,
                "result": None,
            }

        # 4. Return the tool result
        return {
            "status": "success",
            "plan": plan,
            "parsed": parsed,
            "result": result,
        }