"""
agents/grim_trigger.py

Purpose:
- Grim Trigger for graded action space.
- Start fully cooperative (action 0).
- If the *observed group mean cooperation* ever drops below a threshold,
  switch to permanent defection (action M-1).

Notes:
- Uses obs.coop_mean (per-player mean observed cooperation over a window)
  and/or obs.r_obs_prev if available.
- In N-player setting, we use global mean observed cooperation as trigger.
"""

from __future__ import annotations

from typing import Tuple

from game_engine.env.types import AgentMeta, Observation
from .base import BaseAgent


class GrimTrigger(BaseAgent):
    def __init__(self, name: str = "grim", theta: float | None = None):
        super().__init__(name=name)
        self.theta = theta  # if None, will use obs.streak.theta
        self._triggered = False

    def reset(self, seed: int) -> None:
        super().reset(seed)
        self._triggered = False

    def act(self, obs: Observation) -> Tuple[int, AgentMeta]:
        # Choose threshold: prefer explicit theta else use env streak theta
        theta = self.theta if self.theta is not None else float(obs.streak.theta)

        # Global mean observed cooperation (best-effort)
        # Prefer obs.r_obs_prev if it represents rolling observed coop up to t-1
        mean_obs = float(getattr(obs, "r_obs_prev", 0.0))
        if hasattr(obs, "coop_mean") and obs.coop_mean is not None:
            try:
                cm = list(obs.coop_mean)
                if len(cm) > 0:
                    mean_obs = sum(cm) / len(cm)
            except Exception:
                pass

        if not self._triggered and mean_obs < theta:
            self._triggered = True

        a = (obs.M - 1) if self._triggered else 0
        return a, AgentMeta(agent_name=self.name, extra={"triggered": self._triggered, "theta": theta})