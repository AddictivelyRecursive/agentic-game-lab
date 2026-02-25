"""
env/payoff.py

Purpose:
- Computes per-player rewards using TRUE actions (true cooperation levels),
  and current effective parameters (B_eff, C, K).

Return values:
- compute_rewards(true_coop, B_eff, C, K) -> rewards list length N
"""

from __future__ import annotations

from typing import List


def compute_rewards(true_coop: List[float], B_eff: float, C: float, K: float) -> List[float]:
    """Compute rewards for each player.

    u_i = B_eff * avg_{j≠i}(c_j) - C * c_i + K

    Args:
        true_coop: list length N with values in [0,1]
        B_eff: effective benefit parameter for current round
        C: cost coefficient (constant)
        K: offset

    Returns:
        rewards list length N
    """
    N = len(true_coop)
    if N < 2:
        raise ValueError("N must be >= 2 for a meaningful N-player game.")
    if B_eff < 0:
        raise ValueError("B_eff must be >= 0.")
    if C < 0:
        raise ValueError("C must be >= 0.")

    total = sum(true_coop)
    rewards: List[float] = []
    for i, ci in enumerate(true_coop):
        if not (0.0 <= ci <= 1.0):
            raise ValueError("true_coop values must be in [0,1].")
        avg_others = (total - ci) / (N - 1)
        rewards.append(B_eff * avg_others - C * ci + K)
    return rewards