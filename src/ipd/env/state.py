"""
env/state.py

Purpose:
- Holds mutable episode state and rolling statistics.
- Does not implement game logic; simulator orchestrates updates.

Return values:
- StateManager holds fields used by simulator; methods are small and safe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List
import random

from .types import EnvConfig
from .payoff_dynamics import RollingMean


@dataclass
class StateManager:
    """Mutable state for an episode."""
    config: EnvConfig
    rng: random.Random = field(init=False)

    # Time index (1..T). We'll store current round t (next to play).
    t: int = 1

    # Base parameter B_t used for q3 drift, updated after each round.
    B_base: float = 0.0

    # Global streak counter s_{t-1} (up to previous round).
    streak: int = 0

    # Rolling mean of observed global cooperation mean (q3-global driver).
    r_obs_roll: RollingMean = field(init=False)
    r_obs_current: float = 0.0  # rolling value up to previous round (t-1)

    # Histories store actions (ints) per player.
    true_history: List[List[int]] = field(default_factory=list)
    obs_history: List[List[int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        cfg = self.config
        if cfg.N < 2:
            raise ValueError("EnvConfig.N must be >= 2.")
        if cfg.M < 2:
            raise ValueError("EnvConfig.M must be >= 2.")
        if cfg.T <= 0:
            raise ValueError("EnvConfig.T must be > 0.")
        if not (0.0 <= cfg.p_perception <= 1.0):
            raise ValueError("p_perception must be in [0,1].")

        self.rng = random.Random(cfg.seed)
        self.B_base = cfg.payoff.B0
        self.r_obs_roll = RollingMean(cfg.drift.window_w)
        self.r_obs_current = 0.0

        # Initialize empty histories (N lists)
        self.true_history = [[] for _ in range(cfg.N)]
        self.obs_history = [[] for _ in range(cfg.N)]

    def append_actions(self, true_actions: List[int], obs_actions: List[int]) -> None:
        """Append current round actions into histories.

        Args:
            true_actions: length N
            obs_actions: length N
        """
        if len(true_actions) != self.config.N or len(obs_actions) != self.config.N:
            raise ValueError("Action length mismatch with N.")
        for i in range(self.config.N):
            self.true_history[i].append(true_actions[i])
            self.obs_history[i].append(obs_actions[i])