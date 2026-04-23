import json
from ..node import Node


class N6_DecisionPolicy(Node):
    """
    LLM-first decision policy.

    No opponent labels.
    No pre-ranked candidate list.
    The model reasons directly from raw observed history, payoff context,
    action semantics, and derived numeric features.
    """

    def __init__(self):
        super().__init__("N6_DecisionPolicy")

    def run(self, state, context):
        agent_id = state["agent_id"]
        round_num = state["round"]
        M = state["M"]

        base_prompt = context["prompts"]["base_system"]
        decision_prompt = context["prompts"]["decision_policy_system"]

        payload = {
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
            "instruction": {
                "task": "Choose one action to maximize long-term average reward in the repeated game.",
                "requirements": [
                    "Infer patterns directly from raw numeric evidence and observed history.",
                    "Use payoff parameters explicitly.",
                    "Use action semantics explicitly.",
                    "Do not assign categorical labels or personalities to opponents.",
                    "Return JSON only.",
                ],
                "output_schema": {
                    "a": "integer in [0, M-1]",
                    "reason": "brief strategic justification grounded in payoff + history",
                    "confidence": "number in [0,1]",
                },
            },
        }

        user_prompt = json.dumps(payload, ensure_ascii=False, indent=2)
        full_prompt = base_prompt + "\n" + decision_prompt

        try:
            response = context["llm_client"].generate(
                system_prompt=full_prompt,
                user_prompt=user_prompt,
            )
            state.setdefault("llm_raw_outputs", {})
            state["llm_raw_outputs"]["N6"] = response

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
            state["llm_raw_outputs"]["N6_error"] = str(e)
            state["decision"] = None

        return state

    def _safe_parse_json(self, text: str) -> dict:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ValueError("No JSON object found in LLM output.")
        return json.loads(text[start:end + 1])