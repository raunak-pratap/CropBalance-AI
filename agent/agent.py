from typing import Dict

from agent.planner import plan_request
from agent.parser import parse_request
from agent.executor import execute_plan
from agent.verifier import verify_result
from agent.response import build_response
from agent.context.memory import ContextStore


class CropBalanceAgent:
    """
    CropBalance Agent v0.2

    Deterministic agent:
        request → parse → plan → execute → result
    """

    def __init__(self):
        self.pending_request = None
        self.context_store = ContextStore()

    def _complete_pending_request(
        self,
        request: str,
        image_path: str | None = None,
        crop: str | None = None,
        state: str | None = None,
    ) -> Dict | None:
        """Try to complete a previously incomplete request."""

        if not self.pending_request:
            return None

        parsed = parse_request(request)

        pending = self.pending_request

        resolved_crop = pending.get("crop") or parsed.get("crop") or crop
        resolved_state = pending.get("state") or parsed.get("state") or state

        if pending["intent"] == "price_prediction":
            if resolved_crop and resolved_state:
                self.pending_request = None

                return {
                    "intent": "price_prediction",
                    "intents": ["price_prediction"],
                    "crop": resolved_crop,
                    "state": resolved_state,
                }

        if pending["intent"] == "disease_detection":
            if image_path:
                self.pending_request = None

                return {
                    "intent": "disease_detection",
                    "intents": [
                        "disease_detection",
                        "agriculture_advice",
                    ],
                    "crop": resolved_crop,
                    "state": resolved_state,
                }

        return None

    def _update_farmer_context(
        self,
        farmer_id: str | None,
        crop: str | None = None,
        state: str | None = None,
    ) -> None:
        if not farmer_id:
            return

        context = self.context_store.get(farmer_id)

        if context is None:
            from agent.context.models import FarmerContext

            context = FarmerContext(
                farmer_id=farmer_id,
                state=state,
                current_crop=crop,
            )

            self.context_store.save(context)
            return

        self.context_store.update(
            farmer_id,
            current_crop=crop,
            state=state,
        )

    def run(
        self,
        request: str,
        image_path: str | None = None,
        crop: str | None = None,
        state: str | None = None,
        farmer_id: str | None = None,
    ) -> Dict:

        farmer_context = None

        if farmer_id:
            farmer_context = self.context_store.get(farmer_id)

        # 1. Try to complete a previous request first
        completed = self._complete_pending_request(
            request=request,
            image_path=image_path,
            crop=crop,
            state=state,
        )

        if completed:
            parsed = completed
        else:
            parsed = parse_request(request)

            if (
                self.pending_request
                and parsed.get("intent") != self.pending_request.get("intent")
                and parsed.get("intent") != "unknown"
            ):
                self.pending_request = None

        # 2. Resolve missing values from farmer context
        resolved_crop = parsed.get("crop") or crop
        resolved_state = parsed.get("state") or state

        if farmer_context:
            resolved_crop = resolved_crop or farmer_context.current_crop
            resolved_state = resolved_state or farmer_context.state

        if resolved_crop:
            parsed["crop"] = resolved_crop

        if resolved_state:
            parsed["state"] = resolved_state

        crop = resolved_crop
        state = resolved_state

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
        errors = execution.get("errors", {})

        all_missing = bool(errors) and all(
            error.get("status") == "missing_input"
            for error in errors.values()
        )

        if execution["status"] == "error" and all_missing:

            if parsed["intent"] == "disease_detection":
                self.pending_request = {
                    "intent": parsed["intent"],
                    "intents": parsed.get("intents", []),
                    "crop": crop or parsed.get("crop"),
                    "state": state or parsed.get("state"),
                    "missing": ["image"],
                }

                message = (
                    "📷 Please upload a clear image of the affected crop leaf. "
                    "I need the image to identify the disease before providing "
                    "disease-specific treatment advice."
                )

            else:
                message = next(
                    (
                        error.get("message")
                        for error in errors.values()
                        if error.get("status") == "missing_input"
                    ),
                    "Please provide the missing information.",
                )

            if parsed["intent"] == "price_prediction":
                self.pending_request = {
                    "intent": parsed["intent"],
                    "intents": parsed.get("intents", []),
                    "crop": crop or parsed.get("crop"),
                    "state": state or parsed.get("state"),
                    "missing": [
                        missing
                        for error in errors.values()
                        for missing in error.get("missing", [])
                    ],
                }

            return {
                "status": "missing_input",
                "plan": plan,
                "parsed": parsed,
                "result": execution.get("result", {}),
                "errors": errors,
                "message": message,
            }

        if execution["status"] == "error":
            return {
                "status": "error",
                "plan": plan,
                "parsed": parsed,
                "result": execution.get("result", {}),
                "errors": errors,
            }

        # 7. Handle missing input from a single-tool request

        if execution["status"] == "missing_input":

            if parsed["intent"] in [
                "price_prediction",
                "disease_detection",
            ]:
                self.pending_request = {
                    "intent": parsed["intent"],
                    "intents": parsed.get("intents", []),
                    "crop": crop or parsed.get("crop"),
                    "state": state or parsed.get("state"),
                    "missing": execution.get("missing", []),
                }

            return {
                "status": "missing_input",
                "plan": plan,
                "parsed": parsed,
                "result": execution.get("result", {}),
                "missing": execution.get("missing", []),
                "message": execution.get(
                    "message",
                    "Please provide the missing information.",
                ),
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

        self._update_farmer_context(
            farmer_id=farmer_id,
            crop=parsed.get("crop"),
            state=parsed.get("state"),
        )

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
