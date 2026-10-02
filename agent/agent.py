from typing import Dict

from agent.planner import plan_request
from agent.parser import parse_request
from agent.executor import execute_plan
from agent.verifier import verify_result
from agent.response import build_response


class CropBalanceAgent:
    """
    CropBalance Agent v0.2

    Deterministic agent:
        request → parse → plan → execute → result
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
                    "intents": ["price_prediction"],
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

        # 4. Check whether we understand the request
        if not plan.get("tools"):
            return {
                "status": "unsupported",
                "plan": plan,
                "parsed": parsed,
                "result": None,
            }

        # 5. Execute selected tool(s)
        execution = execute_plan(
            plan=plan,
            parsed=parsed,
            image_path=image_path,
            crop=crop,
            state=state,
        )

                # 6. Handle execution failures
        if execution["status"] == "error":
            return {
                "status": "error",
                "plan": plan,
                "parsed": parsed,
                "result": execution.get("result", {}),
                "errors": execution.get("errors", {}),
            }

        # 7. Handle missing input from a single-tool request
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
                "result": execution.get("result"),
                "missing": execution.get("missing", []),
                "message": execution.get("message"),
            }

        # 8. Verify every successful tool result
        verified_results = {}
        verification_errors = {}

        for tool, result in execution.get("result", {}).items():
            verification = verify_result(
                tool=tool,
                result=result,
            )

            if verification["valid"]:
                verified_results[tool] = {
                    "result": result,
                    "verification": verification,
                }
            else:
                verification_errors[tool] = verification

        # 9. Handle partial execution
        if execution["status"] == "partial_success":
            return {
                "status": "partial_success",
                "plan": plan,
                "parsed": parsed,
                "result": verified_results,
                "errors": execution.get("errors", {}),
                "verification_errors": verification_errors,
            }

        # 10. Handle verification failure
        if verification_errors and not verified_results:
            return {
                "status": "verification_failed",
                "plan": plan,
                "parsed": parsed,
                "result": {},
                "verification_errors": verification_errors,
            }

        # 11. Everything executed and verified successfully
        response = build_response(
        verified_results=verified_results,
        status="success",
        errors=execution.get("errors", {}),
        )

        return {
            "status": "success",
            "plan": plan,
            "parsed": parsed,
            "result": verified_results,
            "response": response,
            "errors": execution.get("errors", {}),
            "verification_errors": verification_errors,
        }

        