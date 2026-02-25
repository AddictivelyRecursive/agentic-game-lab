from ..node import Node
import numpy as np


class N2_ComputePayoff(Node):
    """
    Compute effective payoff parameters for this round.

    Computes:
    - B_eff using streak rule
    - Stores payoff constants needed for EU computation
    """

    def __init__(self):
        super().__init__("N2_ComputePayoff")

    def run(self, state, context):

        payoff = state["payoff"]
        streak_rule = state["streak_rule"]
        streak_effect = state["streak_effect"]

        B_base = payoff["B_base"]
        C = payoff["C"]
        K = payoff["K"]

        streak_prev = streak_rule["streak_prev"]
        lambda_ = streak_rule["lambda"]
        tau = streak_rule["tau"]

        # B_eff = B_base * (1 + lambda * tanh(streak_prev / tau))
        B_eff = B_base * (1 + lambda_ * np.tanh(streak_prev / tau))

        state["B_eff"] = float(B_eff)
        state["C"] = C
        state["K"] = K

        return state