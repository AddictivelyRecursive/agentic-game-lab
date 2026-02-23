"""
agents/baselines/always.py

Purpose:
- AlwaysCooperate: returns action 0 (max cooperation by convention).
- AlwaysDefect: returns action M-1 (min cooperation).

Return values:
- act(obs) returns (action:int, AgentMeta).
"""

from __future__ import annotations

from typing import Tuple

from ipd.env.types import AgentMeta, Observation
from .base import BaseAgent


class AlwaysCooperate(BaseAgent):
    """Always plays max cooperation action a=0."""
    def __init__(self, name: str = "always_cooperate"):
        super().__init__(name=name)

    def act(self, obs: Observation) -> Tuple[int, AgentMeta]:
        return 0, AgentMeta(agent_name=self.name)


class AlwaysDefect(BaseAgent):
    """Always plays min cooperation action a=M-1."""
    def __init__(self, name: str = "always_defect"):
        super().__init__(name=name)

    def act(self, obs: Observation) -> Tuple[int, AgentMeta]:
        return obs.M - 1, AgentMeta(agent_name=self.name)