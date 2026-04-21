"""
agents/urnd_agent.py

Purpose:
- UnfairRandomAgent for graded discrete action spaces.
- Generalizes the binary URND opponent to any M >= 2.

Semantics:
- Lower action index = more cooperation.
- For M actions, we assume the implied cooperation level of action a is:
      coop(a) = 1 - a / (M - 1)
- p_cooperate is interpreted as the target expected cooperation level in [0, 1].

Examples:
- M=2:
    action 0 has coop 1.0
    action 1 has coop 0.0
    => identical to binary unfair-random
- M=5:
    actions correspond to coop levels [1.0, 0.75, 0.5, 0.25, 0.0]
    p_cooperate=0.6 produces a random mixture whose expected cooperation is 0.6
"""

from __future__ import annotations

import math
from typing import List, Tuple

from game_engine.env.types import AgentMeta, Observation

from .base import BaseAgent


class UnfairRandomAgent(BaseAgent):
    """
    Graded unfair-random baseline.

    Chooses actions in {0, 1, ..., M-1} so that the expected cooperation level
    matches p_cooperate in [0, 1], under the repo convention:
        lower index = more cooperation
    """

    def __init__(
        self,
        name: str = "urnd",
        p_cooperate: float = 0.5,
    ):
        super().__init__(name=name)
        if not (0.0 <= float(p_cooperate) <= 1.0):
            raise ValueError(f"p_cooperate must be in [0,1], got {p_cooperate!r}")
        self.p_cooperate = float(p_cooperate)

    @staticmethod
    def _action_coop_values(M: int) -> List[float]:
        if M <= 1:
            return [1.0]
        return [1.0 - (a / (M - 1)) for a in range(M)]

    @staticmethod
    def _distribution_for_target_coop(M: int, p_cooperate: float) -> List[float]:
        """
        Return a distribution over actions 0..M-1 whose expected cooperation
        equals p_cooperate under the linear action semantics:
            coop(a) = 1 - a/(M-1)

        Construction:
        - Convert desired cooperation to a continuous target index x.
        - Mix only between floor(x) and ceil(x), which is the minimum-entropy
          discrete distribution achieving that expected cooperation.
        """
        if M <= 1:
            return [1.0]

        p = max(0.0, min(1.0, float(p_cooperate)))

        # Continuous target index:
        # p=1.0 -> x=0 (fully cooperative)
        # p=0.0 -> x=M-1 (fully defective)
        x = (1.0 - p) * (M - 1)

        lo = int(math.floor(x))
        hi = int(math.ceil(x))

        probs = [0.0] * M

        if lo == hi:
            probs[lo] = 1.0
            return probs

        frac = x - lo
        probs[lo] = 1.0 - frac
        probs[hi] = frac
        return probs

    def act(self, obs: Observation) -> Tuple[int, AgentMeta]:
        assert self.rng is not None, "Call reset() before act()."

        M = int(obs.M)
        if M <= 0:
            raise ValueError(f"Invalid action space size M={M}")

        probs = self._distribution_for_target_coop(M, self.p_cooperate)

        # Sample from categorical distribution
        u = self.rng.random()
        cum = 0.0
        action = M - 1
        for i, p in enumerate(probs):
            cum += p
            if u <= cum:
                action = i
                break

        coop_values = self._action_coop_values(M)
        expected_coop = sum(p * c for p, c in zip(probs, coop_values))

        return action, AgentMeta(
            agent_name=self.name,
            extra={
                "agent_type": "UnfairRandomAgent",
                "p_cooperate": self.p_cooperate,
                "M": M,
                "action_probs": probs,
                "expected_coop": expected_coop,
                "sampled_action": action,
            },
        )