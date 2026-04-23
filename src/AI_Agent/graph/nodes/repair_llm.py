import json
from ..node import Node


class N8_RepairOrFinalize(Node):
    """
    Repair invalid LLM output using the same raw-evidence context.

    No dependency on candidates, strategy_summary, or opponent forecasts.
    """

    def __init__(self):
        super().__init__("N8_RepairOrFinalize")

    def run(self, state, context):
        error = state.get("validation_error", "Invalid decision")
        M = state["M"]
        agent_id = state["agent_id"]
        round_num = state["round"]

        base_prompt = context["prompts"]["base_system"]
        repair_prompt = context["prompts"]["repair_system"]

        prev_raw = ""
        if "llm_raw_outputs" in state and "N6" in state["llm_raw_outputs"]:
            prev_raw = state["llm_raw_outputs"]["N6"]

        payload = {
            "task": "Repair the previous invalid decision and return valid JSON only.",
            "validation_error": error,
            "previous_invalid_output": prev_raw,
            "round": round_num,
            "agent_id": agent_id,
            "valid_actions": {
                "min": 0,
                "max": M - 1,
                "note": "Lower action index means more cooperation.",
            },
            "raw_turn_input": state.get("raw_input", {}),
            "game_parameters": state.get("game_parameters", {}),
            "payoff_context": {
                "B_eff": state.get("B_eff"),
                "C": state.get("C"),
                "K": state.get("K"),
                "formula": state.get("payoff_context", {}).get(
                    "formula",
                    "u_i = B_eff * avg_other_coop(true) - C * own_coop(true) + K",
                ),
                "details": state.get("payoff_context", {}),
            },
            "noise_context": state.get("noise_model", {}),
            "observation_context": state.get("info", {}),
            "derived_features": state.get("features", {}),
            "output_schema": {
                "a": "integer in [0, M-1]",
                "reason": "brief strategic justification grounded in payoff + history",
                "confidence": "number in [0,1]",
            },
        }

        user_prompt = json.dumps(payload, ensure_ascii=False, indent=2)
        full_prompt = base_prompt + "\n" + repair_prompt

        try:
            response = context["llm_client"].generate(
                system_prompt=full_prompt,
                user_prompt=user_prompt,
            )
            state.setdefault("llm_raw_outputs", {})
            state["llm_raw_outputs"]["N8"] = response

            data = self._safe_parse_json(response)

            decision = {}
            if "a" in data:
                decision["a"] = data["a"]
            if "reason" in data:
                decision["reason"] = data["reason"]
            if "confidence" in data:
                decision["confidence"] = data["confidence"]

            state["decision"] = decision if decision else None

        except Exception as e:
            state.setdefault("llm_raw_outputs", {})
            state["llm_raw_outputs"]["N8_error"] = str(e)
            state["decision"] = None

        return state

    def _safe_parse_json(self, text: str) -> dict:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ValueError("No JSON object found in repaired LLM output.")
        return json.loads(text[start:end + 1])