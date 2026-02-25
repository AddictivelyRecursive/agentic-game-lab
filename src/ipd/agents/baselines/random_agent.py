"""
agents/baselines/random_agent.py

Purpose:
- RandomAgent: chooses a uniformly random action in [0, M-1].

Return values:
- act(obs) returns (action:int, AgentMeta).
"""

from __future__ import annotations

from typing import Tuple

from ipd.env.types import AgentMeta, Observation
from .base import BaseAgent


class RandomAgent(BaseAgent):
    """Uniform random policy baseline."""
    def __init__(self, name: str = "random"):
        super().__init__(name=name)

    def act(self, obs: Observation) -> Tuple[int, AgentMeta]:
        assert self.rng is not None, "Call reset() before act()."
        a = self.rng.randrange(obs.M)
        return a, AgentMeta(agent_name=self.name)