from __future__ import annotations

import json
from typing import Any, Dict, List

from ..node import Node


class N6_DecisionPolicy(Node):
    """
    Final LLM policy head.

    Key change:
    The model now receives BOTH:
    1) raw game context (payoff / action semantics / history / rules)
    2) derived strategic summaries and ranked candidates

    This preserves strategic scaffolding without hiding the actual game.
    """

    def __init__(self):
        super().__init__("N6_DecisionPolicy")

    @staticmethod
    def _compact_history(history: List[List[int]]) -> List[List[int]]:
        if not isinstance(history, list):
            return []
        out: List[List[int]] = []
        for row in history:
            if isinstance(row, list):
                out.append(row[:])
            else:
                out.append([])
        return out

    @staticmethod
    def _make_action_table(index_to_coop: List[float]) -> List[Dict[str, Any]]:
        table: List[Dict[str, Any]] = []
        for idx, coop in enumerate(index_to_coop):
            table.append(
                {
                    "action": int(idx),
                    "true_cooperation_level": float(round(float(coop), 4)),
                }
            )
        return table

    def run(self, state: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        candidates = state["candidates"]
        agent_id = int(state["agent_id"])
        round_num = int(state["round"])
        M = int(state["M"])

        base_prompt = context["prompts"]["base_system"]
        decision_prompt = context["prompts"]["decision_policy_system"]

        top_k = min(max(3, M), len(candidates))
        top_candidates = candidates[:top_k]

        strategy_summary = state.get("strategy_summary", {})
        opponent_style_summary = state.get("opponent_style_summary", {})

        info = state.get("info", {}) or {}
        observed_history_last_k = self._compact_history(
            info.get("observed_history_last_k", [])
        )

        raw_game_context = {
            "round": round_num,
            "agent_id": agent_id,
            "N": int(state["N"]),
            "M": M,
            "valid_actions": list(range(M)),
            "perception_noise_p": float(state.get("p", 0.0)),
            "action_semantics": {
                "index_to_true_cooperation": self._make_action_table(
                    state.get("index_to_coop", [])
                ),
                "note": "Lower action index means more cooperation.",
            },
            "payoff": state.get("payoff_context", {}),
            "streak_rule": state.get("streak_rule", {}),
            "streak_effect_on_payoff": state.get("streak_effect", {}),
            "drift_rule": info.get("drift_rule", {}),
        }

        observation_context = {
            "observed_history_last_k": observed_history_last_k,
            "recent_observed_actions_by_player": [
                row[0] if isinstance(row, list) and len(row) > 0 else None
                for row in observed_history_last_k
            ],
            "rolling_observed_mean_cooperation_prev": info.get("r_obs_prev"),
            "player_mean_observed_cooperation": info.get("player_mean_observed_coop"),
            "action_frequency_matrix": info.get("action_freq_NxM"),
        }

        compact_candidates = []
        for c in top_candidates:
            compact_candidates.append(
                {
                    "action": int(c["action"]),
                    "strategic_score": round(float(c["strategic_score"]), 4),
                    "own_true_coop": round(float(c["own_true_coop"]), 4),
                    "target_coop": round(float(c["target_coop"]), 4),
                    "distance_to_target": round(float(c["distance_to_target"]), 4),
                    "reciprocity_score": round(float(c.get("reciprocity_score", 0.0)), 4),
                    "normalized_eu": round(float(c.get("normalized_eu", 0.0)), 4),
                }
            )

        user_payload = {
            "raw_game_context": raw_game_context,
            "observation_context": observation_context,
            "opponent_behavior_summary": opponent_style_summary,
            "strategic_summary": strategy_summary,
            "top_candidate_actions": compact_candidates,
            "instruction": {
                "objective": (
                    "Choose one action for a repeated strategic game using the FULL "
                    "game context. Explicitly account for payoff incentives, action "
                    "semantics, recent observed behavior, and longer-run strategic effects."
                ),
                "decision_rule": (
                    "Do not ignore the payoff parameters. Use them together with repeated-game "
                    "considerations. The payoff formula and current B_effective are part of the task."
                ),
                "output_schema": {
                    "a": "integer action in [0, M-1]",
                    "reason": "brief strategic justification grounded in payoff + history",
                    "confidence": "number in [0,1]",
                },
                "constraints": [
                    "Return JSON only.",
                    "Be concise.",
                    "Reason must be strategic and grounded in the provided game context.",
                    "Do not output markdown.",
                ],
            },
        }

        full_prompt = base_prompt + "\n" + decision_prompt
        user_prompt = json.dumps(user_payload, ensure_ascii=False, indent=2)

        try:
            try:
                response = context["llm_client"].generate(
                    system_prompt=full_prompt,
                    user_prompt=user_prompt,
                    temperature=0.4,
                )
            except TypeError:
                response = context["llm_client"].generate(
                    system_prompt=full_prompt,
                    user_prompt=user_prompt,
                )

            state.setdefault("llm_raw_outputs", {})
            state["llm_raw_outputs"]["N6"] = response

            data = self._safe_parse_json(response)

            decision: Dict[str, Any] = {}
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
                        return json.loads(text[start : i + 1])
        raise ValueError("No balanced JSON object found in LLM output.")