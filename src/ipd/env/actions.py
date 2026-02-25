"""
env/actions.py

Purpose:
- Defines mapping from discrete action index a in [0, M-1] to cooperation level c in [0,1].
- Provides helpers used by payoff, streaks, and metrics.

Return values:
- levels_for_M(M) returns a list[float] of length M.
- action_to_coop(a, levels) returns float in [0,1].
"""

from __future__ import annotations

from typing import List


def levels_for_M(M: int) -> List[float]:
    """Return cooperation levels for an M-action graded cooperation space.

    Convention:
    - a=0 corresponds to maximum cooperation (1.0)
    - a=M-1 corresponds to minimum cooperation (0.0)
    - linearly spaced levels between.

    This matches your examples:
    - M=3 -> [1.0, 0.5, 0.0]
    - M=5 -> [1.0, 0.75, 0.5, 0.25, 0.0]
    """
    if M < 2:
        raise ValueError("M must be >= 2.")
    return [1.0 - (i / (M - 1)) for i in range(M)]


def action_to_coop(a: int, coop_levels: List[float]) -> float:
    """Map action index to cooperation level.

    Args:
        a: integer action in [0, M-1]
        coop_levels: list returned by levels_for_M(M)

    Returns:
        cooperation level in [0,1]
    """
    if a < 0 or a >= len(coop_levels):
        raise ValueError(f"Invalid action {a} for M={len(coop_levels)}")
    return coop_levels[a]


def actions_to_coop(actions: List[int], coop_levels: List[float]) -> List[float]:
    """Vectorized helper: map list of actions to cooperation levels."""
    return [action_to_coop(a, coop_levels) for a in actions]