from typing import Dict

from agent.planner import plan_request
from agent.parser import parse_request
from agent.executor import execute_plan



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
        execution = execute_plan(
            plan=plan,
            parsed=parsed,
            image_path=image_path,
            crop=crop,
            state=state,
        )

        if execution["status"] == "missing_input":
            if parsed["intent"] == "price_prediction":
                self.pending_request = {
                    "intent": parsed["intent"],
                    "crop": crop or parsed.get("crop"),
                    "state": state or parsed.get("state"),
                    "missing": execution["missing"],
                }

            return {
                "status": "missing_input",
                "plan": plan,
                "parsed": parsed,
                "result": None,
                "missing": execution["missing"],
                "message": execution["message"],
            }

        if execution["status"] == "unsupported":
            return {
                "status": "unsupported",
                "plan": plan,
                "parsed": parsed,
                "result": None,
                "message": execution.get("message"),
            }

        # 4. Return the tool result
        return {
            "status": "success",
            "plan": plan,
            "parsed": parsed,
            "result": execution["result"],
        }