"""
agents/threshold_public_goods.py

Purpose:
- Threshold Public Goods strategy:
  Cooperate if rolling observed cooperation (up to t-1) is at least theta,
  else defect.

For graded actions:
- "Cooperate" -> action 0 (most cooperative)
- "Defect" -> action M-1 (least cooperative)

Threshold theta defaults to obs.streak.theta unless overridden.
"""

from __future__ import annotations

from typing import Optional, Tuple

from game_engine.env.types import AgentMeta, Observation
from .base import BaseAgent


class ThresholdPublicGoods(BaseAgent):
    def __init__(self, name: str = "threshold_pg", theta: Optional[float] = None):
        super().__init__(name=name)
        self.theta = theta

    def act(self, obs: Observation) -> Tuple[int, AgentMeta]:
        theta = self.theta if self.theta is not None else float(obs.streak.theta)
        r_obs_prev = float(getattr(obs, "r_obs_prev", 0.0))

        a = 0 if r_obs_prev >= theta else (obs.M - 1)
        return a, AgentMeta(agent_name=self.name, extra={"theta": theta, "r_obs_prev": r_obs_prev})