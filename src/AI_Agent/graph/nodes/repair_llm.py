import json

from ..node import Node


class N8_RepairOrFinalize(Node):
    """
    Repair node for invalid LLM output.

    Important change:
    - Repair prompt now uses strategic summary / strategic candidates,
      not "expected utility descending" language.
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

        candidates = state.get("candidates", [])
        top_k = min(max(3, M), len(candidates))
        top_candidates = candidates[:top_k]
        strategy_summary = state.get("strategy_summary", {})

        prev_raw = ""
        if "llm_raw_outputs" in state and "N6" in state["llm_raw_outputs"]:
            prev_raw = state["llm_raw_outputs"]["N6"]

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

Your previous output was invalid because: {error}

This is a repeated strategic game.
Choose for long-run performance, not just immediate one-step payoff.
Use the strategic summary and candidates below.

Strategic summary:
{strategy_summary}

Top candidate actions (sorted by strategic_score desc):
{compact_candidates}

Return corrected JSON only:
{{
  "a": <int>,
  "reason": "<brief strategic reason>",
  "confidence": <float between 0 and 1>
}}

Previous raw output (for reference):
{prev_raw}
""".strip()

        full_prompt = base_prompt + "\n" + repair_prompt

        try:
            response = context["llm_client"].generate(
                system_prompt=full_prompt,
                user_prompt=user_prompt,
            )
            state.setdefault("llm_raw_outputs", {})
            state["llm_raw_outputs"]["N8"] = response

            data = self._safe_parse_json(response)

            repaired = {}
            if "a" in data:
                repaired["a"] = data["a"]
            if "reason" in data:
                repaired["reason"] = data["reason"]
            if "confidence" in data:
                repaired["confidence"] = data["confidence"]

            state["decision"] = repaired if repaired else None

        except Exception as e:
            state.setdefault("llm_raw_outputs", {})
            state["llm_raw_outputs"]["N8_error"] = str(e)
            state["decision"] = None

        return state

    def _safe_parse_json(self, text: str) -> dict:
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