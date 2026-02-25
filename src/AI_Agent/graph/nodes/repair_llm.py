import json
from ..node import Node


class N8_RepairOrFinalize(Node):
    """
    Repair node for invalid LLM output.

    Triggered only when N7 marks decision invalid.
    It asks the LLM to output the STRICT JSON schema again.

    Expected output:
      {
        "a": <int>,
        "reason": "<short explanation, <= 2 sentences>",
        "confidence": <float in [0,1]>
      }

    If repair fails, decision remains None and GraphRunner will either retry
    again (up to max_retries) or fall back to N9.
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

        # Help the model choose something sensible by giving top candidates
        candidates = state.get("candidates", [])
        top_k = context.get("top_k_candidates", 5)
        top_candidates = candidates[:top_k]

        prev_raw = ""
        if "llm_raw_outputs" in state and "N6" in state["llm_raw_outputs"]:
            prev_raw = state["llm_raw_outputs"]["N6"]

        user_prompt = f"""
Round: {round_num}
Agent ID: {agent_id}
Valid actions: integers in [0, {M-1}]

Previous output was invalid because:
{error}

Top candidate actions (sorted by expected utility desc):
{top_candidates}

Return corrected JSON only.
Previous raw output (for reference):
{prev_raw}
"""

        full_prompt = base_prompt + "\n" + repair_prompt

        try:
            response = context["llm_client"].generate(
                system_prompt=full_prompt,
                user_prompt=user_prompt
            )

            state.setdefault("llm_raw_outputs", {})
            state["llm_raw_outputs"]["N8"] = response

            data = self._safe_parse_json(response)

            # Keep only schema keys if present
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
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ValueError("No JSON object found in LLM output.")
        return json.loads(text[start:end + 1])