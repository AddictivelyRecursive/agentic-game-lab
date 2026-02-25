import json
from ..node import Node


class N4_OpponentModel(Node):
    """
    LLM-based opponent modeling.

    This node:
    - Uses observed action frequencies
    - Asks LLM to predict probability distribution of next action
      for each opponent (excluding self)
    - Stores structured opponent_action_probs in state

    If LLM fails, falls back to empirical action_freq_NxM.
    """

    def __init__(self):
        super().__init__("N4_OpponentModel")

    def run(self, state, context):

        agent_id = state["agent_id"]
        action_freq = state["info"]["action_freq_NxM"]
        N = state["N"]
        M = state["M"]

        base_prompt = context["prompts"]["base_system"]
        opp_prompt = context["prompts"]["opponent_model_system"]

        user_prompt = f"""
You are given empirical action frequencies of players in a repeated game.

Players: {N}
Actions: {M} (indexed 0 to {M-1})

Empirical frequencies (NxM):
{action_freq}

Your agent_id: {agent_id}

For each opponent (exclude agent_id), output a probability distribution
over actions for next round.

Output JSON:
{{
  "opponent_action_probs": {{
      "<player_id>": [p0, p1, ..., pM-1]
  }}
}}
"""

        full_prompt = base_prompt + "\n" + opp_prompt

        try:
            response = context["llm_client"].generate(
                system_prompt=full_prompt,
                user_prompt=user_prompt
            )

            data = self._safe_parse_json(response)

            state["opponent_action_probs"] = data["opponent_action_probs"]

        except Exception:
            # Fallback to empirical distribution
            fallback = {}
            for i in range(N):
                if i == agent_id:
                    continue
                fallback[str(i)] = action_freq[i]

            state["opponent_action_probs"] = fallback

        return state

    def _safe_parse_json(self, text):
        """
        Extract JSON object from LLM output safely.
        """
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("Invalid JSON")
        return json.loads(text[start:end+1])