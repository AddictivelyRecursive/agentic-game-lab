"""
agents/baselines/graded_tft.py

Purpose:
- GradedTFT: generalized tit-for-tat for graded action space.
- Uses last observed round (t-1) and matches the group's mean observed action level.

Interpretation:
- If others were more cooperative last round (lower action index), agent moves cooperative.
- If others were less cooperative (higher action index), agent moves defect-like.

Return values:
- act(obs) returns (action:int, AgentMeta).
"""

from __future__ import annotations

from typing import Tuple

from game_engine.env.types import AgentMeta, Observation
from .base import BaseAgent


class GradedTFT(BaseAgent):
    """Match last-round observed group cooperation (via mean observed action index)."""
    def __init__(self, name: str = "graded_tft"):
        super().__init__(name=name)

    def act(self, obs: Observation) -> Tuple[int, AgentMeta]:
        # obs.observed_history_packed is list[str] of length N.
        # Each string is length <= k, with last char being action at t-1 (if available).
        last_actions = []
        for s in obs.observed_history_packed:
            if len(s) == 0:
                continue
            # digit char -> int action (works because env packing assumes M<=10)
            last_actions.append(int(s[-1]))

        if len(last_actions) == 0:
            # No history yet: start cooperatively
            return 0, AgentMeta(agent_name=self.name)

        mean_a = sum(last_actions) / len(last_actions)
        a = int(round(mean_a))
        a = max(0, min(obs.M - 1, a))
        return a, AgentMeta(agent_name=self.name)