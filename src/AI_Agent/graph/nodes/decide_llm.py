import json

from ..node import Node


class N6_DecisionPolicy(Node):
    """
    LLM-based decision policy.

    Important change:
    - The LLM is no longer shown only an "expected utility descending" shortlist.
    - It now receives a strategic summary + candidates ranked by strategic_score.
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

        top_k = min(max(3, M), len(candidates))
        top_candidates = candidates[:top_k]
        strategy_summary = state.get("strategy_summary", {})

        history = state.get("info", {}).get("observed_history_last_k", [])
        last_observed_actions = []
        if isinstance(history, list):
            for row in history:
                if isinstance(row, list) and len(row) > 0:
                    last_observed_actions.append(row[0])
                else:
                    last_observed_actions.append(None)

        compact_candidates = []
        for c in top_candidates:
            compact_candidates.append(
                {
                    "action": c["action"],
                    "strategic_score": round(float(c["strategic_score"]), 4),
                    "own_true_coop": round(float(c["own_true_coop"]), 4),
                    "target_coop": round(float(c["target_coop"]), 4),
                    "distance_to_target": round(float(c["distance_to_target"]), 4),
                    "EU": round(float(c["EU"]), 4),
                }
            )

        user_prompt = f"""
Round: {round_num}
Agent ID: {agent_id}
Valid actions: integers in [0, {M - 1}]

This is a repeated strategic game.
Choose for long-run performance, not just immediate one-step payoff.
Reciprocate sustained cooperation, punish persistent defection, and forgive when the evidence improves.

Most recent observed actions by player (None means no history): {last_observed_actions}

Strategic summary:
{strategy_summary}

Top candidate actions (sorted by strategic_score desc):
{compact_candidates}

Return JSON only:
{{
  "a": <int>,
  "reason": "<brief strategic reason>",
  "confidence": <float between 0 and 1>
}}
""".strip()

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
        """
        Extract the first balanced JSON object from possibly noisy LLM output.
        """
        start = None
        depth = 0

        for i, ch in enumerate(text):
            if ch == "{":
                if start is None:
                    start = i
                depth += 1
            elif ch == "}":
                if start is not None:
                    depth -= 1
                    if depth == 0:
                        return json.loads(text[start:i + 1])

        raise ValueError("No balanced JSON object found in LLM output.")