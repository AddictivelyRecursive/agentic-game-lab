"""
agents/win_stay_lose_shift.py

Purpose:
- Win-Stay Lose-Shift (Pavlov) for graded action space.
- If last round looked "good" (win), repeat last own action.
- Else shift one step toward defection (or optionally toward cooperation).

Two win criteria are supported:
1) reward_threshold: if last reward >= threshold => win
2) r_obs_prev threshold: if rolling observed cooperation >= theta => win

Implementation detail:
- Uses obs.observed_history_packed for last own action (t-1).
- Uses obs.r_obs_prev and obs.streak.theta for coop-based win signal.
- Reward-based win uses obs.extra if available; otherwise falls back to coop signal.

Because your Observation currently doesn't include last reward directly,
the default is coop-based win using obs.r_obs_prev >= theta.
"""

from __future__ import annotations

from typing import Optional, Tuple

from game_engine.env.types import AgentMeta, Observation
from .base import BaseAgent


class WinStayLoseShift(BaseAgent):
    def __init__(
        self,
        name: str = "wsls",
        *,
        reward_threshold: Optional[float] = None,
        use_reward_if_available: bool = True,
        theta: Optional[float] = None,
        shift_direction: str = "toward_defect",  # "toward_defect" or "toward_coop"
    ):
        super().__init__(name=name)
        self.reward_threshold = reward_threshold
        self.use_reward_if_available = use_reward_if_available
        self.theta = theta
        self.shift_direction = shift_direction

    def act(self, obs: Observation) -> Tuple[int, AgentMeta]:
        # Determine last own action (t-1) from packed history
        last_own: Optional[int] = None
        try:
            s = obs.observed_history_packed[obs.agent_id]
            if isinstance(s, str) and len(s) > 0:
                last_own = int(s[-1])
        except Exception:
            last_own = None

        if last_own is None:
            # No history: start cooperative
            return 0, AgentMeta(agent_name=self.name, extra={"win": None, "reason": "no_history"})

        # Decide "win" signal
        win = None
        reason = "coop_signal"

        # (A) reward-based, only if present in obs (not guaranteed)
        if self.use_reward_if_available and self.reward_threshold is not None:
            last_reward = getattr(obs, "last_reward", None)
            if last_reward is not None:
                win = float(last_reward) >= float(self.reward_threshold)
                reason = "reward_threshold"

        # (B) coop-based fallback (recommended for your current Observation)
        if win is None:
            theta = self.theta if self.theta is not None else float(obs.streak.theta)
            r_obs_prev = float(getattr(obs, "r_obs_prev", 0.0))
            win = r_obs_prev >= theta
            reason = "r_obs_prev_vs_theta"

        if win:
            a = last_own
        else:
            if self.shift_direction == "toward_coop":
                a = max(0, last_own - 1)
            else:
                a = min(obs.M - 1, last_own + 1)

        return a, AgentMeta(agent_name=self.name, extra={"win": win, "reason": reason, "last_own": last_own})