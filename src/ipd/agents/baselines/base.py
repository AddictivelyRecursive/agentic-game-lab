"""
agents/baselines/base.py

Purpose:
- Defines a tiny BaseAgent with common utilities for baseline testing.
- Keeps baseline agents independent of LLM tooling.

Return values:
- Agents implement reset(seed) and act(obs)->(action, AgentMeta).
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Optional, Tuple

from ipd.env.types import AgentMeta, Observation


@dataclass
class BaseAgent:
    """Convenience base for baseline agents."""
    name: str
    rng: Optional[random.Random] = None

    def reset(self, seed: int) -> None:
        """Reset internal RNG."""
        self.rng = random.Random(seed)

    def act(self, obs: Observation) -> Tuple[int, AgentMeta]:
        """Return (action, meta). Must be overridden."""
        raise NotImplementedError