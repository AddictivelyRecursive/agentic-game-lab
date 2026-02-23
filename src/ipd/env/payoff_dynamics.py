"""
env/payoff_dynamics.py

Purpose:
- Computes B_eff from B_base and streak_prev (used for current round payoff).
- Maintains rolling observed cooperation rate r_obs for q3-global drift.
- Updates B_base via bounded drift after each round.

Return values:
- compute_B_eff(B_base, streak_prev, lam, tau) -> B_eff
- update_rolling_r_obs(rolling, obs_coop_mean, w) -> r_obs
- update_B_q3(B_base, r_obs, eta, r_star, B_min, B_max) -> B_next
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Deque
from collections import deque
import math


@dataclass
class RollingMean:
    """Efficient rolling mean over the last w scalar values."""
    w: int
    buf: Deque[float] = field(default_factory=deque)
    s: float = 0.0

    def __post_init__(self) -> None:
        if self.w <= 0:
            raise ValueError("RollingMean window w must be > 0.")

    def update(self, x: float) -> float:
        """Add x, evict oldest if needed, return current mean."""
        self.buf.append(x)
        self.s += x
        if len(self.buf) > self.w:
            self.s -= self.buf.popleft()
        return self.s / len(self.buf)


def compute_B_eff(B_base: float, streak_prev: int, lam: float, tau: float) -> float:
    """Compute effective benefit used in payoff for current round.

    B_eff = B_base * (1 + lam * tanh(streak_prev/tau))

    Uses streak_prev (t-1) to avoid circularity / future leakage.

    Returns:
        B_eff
    """
    if B_base < 0:
        raise ValueError("B_base must be >= 0.")
    if streak_prev < 0:
        raise ValueError("streak_prev must be >= 0.")
    if lam < 0:
        raise ValueError("lam must be >= 0.")
    if tau <= 0:
        raise ValueError("tau must be > 0.")
    g = math.tanh(streak_prev / tau)
    return B_base * (1.0 + lam * g)


def update_B_q3(B_base: float, r_obs: float, eta: float, r_star: float, B_min: float, B_max: float) -> float:
    """Update base benefit B using q3-global drift and clip to bounds.

    B_next = clip(B_base + eta*(r_obs - r_star), [B_min, B_max])

    Returns:
        B_next
    """
    if B_min > B_max:
        raise ValueError("B_min must be <= B_max.")
    if not (0.0 <= r_obs <= 1.0):
        # r_obs is a mean cooperation rate; should lie in [0,1]
        raise ValueError("r_obs must be in [0,1].")
    # eta can be 0; negative eta would invert dynamics (likely undesired)
    if eta < 0:
        raise ValueError("eta must be >= 0.")
    if not (0.0 <= r_star <= 1.0):
        raise ValueError("r_star must be in [0,1].")

    candidate = B_base + eta * (r_obs - r_star)
    return max(B_min, min(B_max, candidate))