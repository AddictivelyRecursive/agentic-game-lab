import json
from ..node import Node


class N6_DecisionPolicy(Node):
    """
    LLM-based decision policy.

    Takes ranked candidates and chooses final action.

    Expected JSON output (preferred):
      {
        "a": <int>,
        "reason": "<short explanation, <= 2 sentences>",
        "confidence": <float in [0,1]>
      }

    Robustness:
    - If LLM returns extra text, we extract the first JSON object.
    - If "reason"/"confidence" missing, we still accept as long as "a" is valid
      (validator checks only "a").
    - Stores raw LLM output in state for analysis.
    """

    def __init__(self):
        super().__init__("N6_DecisionPolicy")

    def run(self, state, context):
        candidates = state["candidates"]
        agent_id = state["agent_id"]
        round_num = state["round"]
        M = state["M"]

        base_prompt = context["prompts"]["base_system"]
        decision_prompt = context["prompts"]["decision_policy_system"]

        # Keep prompt small: send top-K candidates only
        top_k = context.get("top_k_candidates", 5)
        top_candidates = candidates[:top_k]

        user_prompt = f"""
Round: {round_num}
Agent ID: {agent_id}
Valid actions: integers in [0, {M-1}]

Top candidate actions (sorted by expected utility desc):
{top_candidates}

Choose an action to maximize long-term average reward.
"""

        full_prompt = base_prompt + "\n" + decision_prompt

        try:
            response = context["llm_client"].generate(
                system_prompt=full_prompt,
                user_prompt=user_prompt
            )

            state.setdefault("llm_raw_outputs", {})
            state["llm_raw_outputs"]["N6"] = response

            data = self._safe_parse_json(response)

            # Minimal normalization: keep only relevant keys if present
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
        """
        Extract the outermost JSON object from possibly noisy LLM output.
        """
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ValueError("No JSON object found in LLM output.")
        return json.loads(text[start:end + 1])