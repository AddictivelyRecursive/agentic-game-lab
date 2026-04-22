from __future__ import annotations

from typing import Any, Dict

import numpy as np

from ..node import Node


class N2_ComputePayoff(Node):
    """
    Compute effective payoff parameters for this round and expose them
    in a prompt-ready structure for downstream LLM consumption.
    """

    def __init__(self):
        super().__init__("N2_ComputePayoff")

    def run(self, state: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        payoff = state["payoff"]
        streak_rule = state["streak_rule"]

        B_base = float(payoff["B_base"])
        C = float(payoff["C"])
        K = float(payoff["K"])

        streak_prev = int(streak_rule["streak_prev"])
        lambda_ = float(streak_rule["lambda"])
        tau = float(streak_rule["tau"])

        # Effective benefit used this round.
        B_eff = B_base * (1.0 + lambda_ * np.tanh(streak_prev / tau))

        state["B_eff"] = float(B_eff)
        state["C"] = float(C)
        state["K"] = float(K)

        # Prompt-facing structured context.
        state["payoff_context"] = {
            "formula": payoff.get(
                "formula",
                "u_i = B_eff * avg_other_coop(true) - C * own_coop(true) + K",
            ),
            "B_base": float(B_base),
            "B_effective": float(B_eff),
            "C": float(C),
            "K": float(K),
            "streak_prev": streak_prev,
            "lambda": float(lambda_),
            "tau": float(tau),
            "interpretation": {
                "higher_avg_other_coop_increases_payoff": True,
                "higher_own_coop_increases_cost_term": True,
                "current_round_payoff_depends_on_true_cooperation_levels": True,
            },
        }

        return state