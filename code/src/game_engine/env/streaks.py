"""
env/streaks.py

Purpose:
- Maintains a global streak counter based on observed global mean cooperation.
- Provides a bounded streak bonus function g(s) used to modulate B_eff.

Return values:
- update_streak(s_prev, obs_coop, theta) -> s_next
- streak_bonus(s, tau) -> g in [0,1)
"""

from __future__ import annotations

from typing import List
import math


def streak_bonus(streak: int, tau: float) -> float:
    """Bounded streak bonus g(s) = tanh(s/tau).

    Args:
        streak: nonnegative integer
        tau: positive saturation constant

    Returns:
        g in [0, 1)
    """
    if streak < 0:
        raise ValueError("streak must be >= 0.")
    if tau <= 0:
        raise ValueError("tau must be > 0.")
    return math.tanh(streak / tau)


def update_streak(streak_prev: int, obs_coop: List[float], theta: float) -> int:
    """Update global streak based on observed mean cooperation at current round.

    Rule:
    - Let C_obs_mean = mean(obs_coop)
    - If C_obs_mean >= theta: streak_next = streak_prev + 1
    - Else: streak_next = 0

    Args:
        streak_prev: streak up to previous round
        obs_coop: observed cooperation vector at current round (len N, each in [0,1])
        theta: threshold in [0,1]

    Returns:
        streak_next
    """
    if streak_prev < 0:
        raise ValueError("streak_prev must be >= 0.")
    if not (0.0 <= theta <= 1.0):
        raise ValueError("theta must be in [0,1].")
    if len(obs_coop) == 0:
        raise ValueError("obs_coop must be non-empty.")

    mean_obs = sum(obs_coop) / len(obs_coop)
    return (streak_prev + 1) if (mean_obs >= theta) else 0