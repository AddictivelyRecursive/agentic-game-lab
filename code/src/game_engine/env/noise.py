"""
env/noise.py

Purpose:
- Implements q3-global perception noise: each player's true action may be misperceived.
- Noise is applied ONCE per round globally, producing one shared observed action per player.

Return values:
- perceive_actions(true_actions, p, M, rng) -> observed_actions
"""

from __future__ import annotations

from typing import List
import random


def perceive_actions(true_actions: List[int], p: float, M: int, rng: random.Random) -> List[int]:
    """Apply perception noise to a vector of true actions.

    Noise model:
    - With probability p, observed action is replaced by a uniformly random DIFFERENT action.
    - With probability 1-p, observed action equals true action.

    Args:
        true_actions: list of length N, values in [0, M-1]
        p: perception noise in [0,1]
        M: number of actions
        rng: deterministic PRNG

    Returns:
        observed_actions: list of length N, values in [0, M-1]
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError("p_perception must be in [0,1].")
    if M < 2:
        raise ValueError("M must be >= 2.")

    obs: List[int] = []
    for a in true_actions:
        if a < 0 or a >= M:
            raise ValueError(f"Invalid true action {a} for M={M}")
        if rng.random() < p:
            # Choose a different action uniformly.
            choices = list(range(M))
            choices.remove(a)
            obs.append(rng.choice(choices))
        else:
            obs.append(a)
    return obs