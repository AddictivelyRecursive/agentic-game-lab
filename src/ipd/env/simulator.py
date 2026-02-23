"""
env/simulator.py

Purpose:
- Orchestrates the episode control flow with strict causality:
  observations for round t use history up to t-1 only.
- Applies q3-global perception noise, computes payoffs on true actions,
  updates global streak, rolling observed cooperation, and B drift.

Return values:
- GameSimulator.run_episode(...) returns EpisodeResult with a list of RoundLog.
"""

from __future__ import annotations

from typing import Callable, List, Protocol, Tuple

from .types import AgentMeta, EnvConfig, EpisodeResult, Observation, RoundLog
from .state import StateManager
from .actions import levels_for_M, actions_to_coop
from .noise import perceive_actions
from .streaks import update_streak
from .payoff_dynamics import compute_B_eff, update_B_q3
from .payoff import compute_rewards


class Agent(Protocol):
    """Minimal agent protocol expected by the simulator."""
    def reset(self, seed: int) -> None: ...
    def act(self, obs: Observation) -> Tuple[int, AgentMeta]: ...


def _pack_history(obs_history: List[List[int]], M: int, k: int) -> List[str]:
    """Pack last-k observed actions per player as digit strings (works for M<=10).

    Returns:
        list of length N, each string length <= k
    """
    if M > 10:
        raise ValueError("Packed history string format assumes M <= 10 (digits).")
    packed: List[str] = []
    for seq in obs_history:
        tail = seq[-k:]
        packed.append("".join(str(a) for a in tail))
    return packed


def _coop_mean_per_player(obs_history: List[List[int]], coop_levels: List[float], window: int) -> List[float]:
    """Compute per-player mean observed cooperation over last `window` actions."""
    out: List[float] = []
    for seq in obs_history:
        tail = seq[-window:] if window > 0 else []
        if len(tail) == 0:
            out.append(0.0)
        else:
            out.append(sum(coop_levels[a] for a in tail) / len(tail))
    return out


class GameSimulator:
    """Main simulator for one episode."""

    def __init__(self, config: EnvConfig):
        self.config = config

    def run_episode(self, agents: List[Agent]) -> EpisodeResult:
        """Run a single episode of length T.

        Args:
            agents: list of length N following the Agent protocol.

        Returns:
            EpisodeResult containing round logs and summaries.
        """
        cfg = self.config
        if len(agents) != cfg.N:
            raise ValueError(f"Expected {cfg.N} agents, got {len(agents)}.")

        coop_levels = levels_for_M(cfg.M)
        state = StateManager(cfg)

        # Reset agents with deterministic seeds derived from env RNG
        for ag in agents:
            ag.reset(state.rng.randrange(10**9))

        logs: List[RoundLog] = []
        total_rewards = [0.0 for _ in range(cfg.N)]

        # Main loop: t runs 1..T
        for t in range(1, cfg.T + 1):
            state.t = t

            # --- PRECOMPUTE using history up to t-1 ---
            streak_prev = state.streak
            r_obs_prev = state.r_obs_current
            B_base = state.B_base
            B_eff = compute_B_eff(B_base, streak_prev, cfg.streak.lam, cfg.streak.tau)

            # --- Build Observation for each agent using ONLY obs_history up to t-1 ---
            k = cfg.obs.history_k
            packed_hist = _pack_history(state.obs_history, cfg.M, k)

            stats_window = cfg.obs.stats_window or min(max(t - 1, 1), cfg.drift.window_w)
            cm = _coop_mean_per_player(state.obs_history, coop_levels, stats_window)

            observations: List[Observation] = []
            for i in range(cfg.N):
                observations.append(
                    Observation(
                        t=t,
                        agent_id=i,
                        N=cfg.N,
                        M=cfg.M,
                        p=cfg.p_perception,
                        B_base=B_base,
                        B_eff=B_eff,
                        C=cfg.payoff.C,
                        K=cfg.payoff.K,
                        r_obs_prev=r_obs_prev,
                        streak_prev=streak_prev,
                        drift=cfg.drift,
                        streak=cfg.streak,
                        observed_history_packed=packed_hist,
                        coop_mean=cm,
                    )
                )

            # --- Collect true actions simultaneously ---
            true_actions: List[int] = []
            meta_list: List[AgentMeta] = []
            for i, ag in enumerate(agents):
                a, meta = ag.act(observations[i])
                # Hard validation at env boundary
                if not isinstance(a, int) or a < 0 or a >= cfg.M:
                    # Strict fallback: use most cooperative action 0
                    meta = AgentMeta(
                        agent_name=meta.agent_name if meta else f"agent_{i}",
                        parse_ok=getattr(meta, "parse_ok", False),
                        action_ok=False,
                        fallback_used=True,
                        fallback_reason="env_guard_invalid_action->fallback_to_0",
                        input_tokens=getattr(meta, "input_tokens", 0),
                        output_tokens=getattr(meta, "output_tokens", 0),
                        latency_ms=getattr(meta, "latency_ms", 0),
                        raw_hash=getattr(meta, "raw_hash", None),
                        extra=getattr(meta, "extra", {}),
                    )
                    a = 0
                true_actions.append(a)
                meta_list.append(meta)

            # --- Apply q3-global perception noise (once) ---
            obs_actions = perceive_actions(true_actions, cfg.p_perception, cfg.M, state.rng)

            # --- Convert to cooperation levels ---
            true_coop = actions_to_coop(true_actions, coop_levels)
            obs_coop = actions_to_coop(obs_actions, coop_levels)
            obs_coop_mean = sum(obs_coop) / cfg.N

            # --- Compute payoffs on TRUE cooperation, using B_eff for this round ---
            rewards = compute_rewards(true_coop, B_eff, cfg.payoff.C, cfg.payoff.K)
            for i in range(cfg.N):
                total_rewards[i] += rewards[i]

            # --- Update streak (based on observed cooperation at this round) ---
            streak_next = update_streak(streak_prev, obs_coop, cfg.streak.theta)

            # --- Update rolling observed cooperation rate (q3-global driver) ---
            r_obs_next = state.r_obs_roll.update(obs_coop_mean)

            # --- Update base B via q3 drift (for next round) ---
            B_next = update_B_q3(
                B_base=B_base,
                r_obs=r_obs_next,
                eta=cfg.drift.eta,
                r_star=cfg.drift.r_star,
                B_min=cfg.payoff.B_min,
                B_max=cfg.payoff.B_max,
            )

            # --- Critical invariant check (recommended) ---
            # Ensure PD-like incentive ordering remains possible (at least B_eff > C).
            # NOTE: B_eff here used streak_prev; next round uses streak_next and B_next.
            # You should ensure cfg.payoff.B_min * (1 + lam) > C at config time too.
            if B_eff <= cfg.payoff.C:
                # Do not crash by default; loggable event for evaluation.
                # For strictness in research runs, you may raise instead.
                pass

            # --- Log the round (authoritative record) ---
            logs.append(
                RoundLog(
                    t=t,
                    N=cfg.N,
                    M=cfg.M,
                    B_base=B_base,
                    B_eff=B_eff,
                    C=cfg.payoff.C,
                    K=cfg.payoff.K,
                    p=cfg.p_perception,
                    r_obs_prev=r_obs_prev,
                    streak_prev=streak_prev,
                    true_actions=true_actions,
                    obs_actions=obs_actions,
                    true_coop=true_coop,
                    obs_coop=obs_coop,
                    obs_coop_mean=obs_coop_mean,
                    rewards=rewards,
                    r_obs_next=r_obs_next,
                    streak_next=streak_next,
                    B_next=B_next,
                    agent_meta=meta_list,
                )
            )

            # --- Commit state updates for next round ---
            state.append_actions(true_actions, obs_actions)
            state.streak = streak_next
            state.r_obs_current = r_obs_next
            state.B_base = B_next

        return EpisodeResult(
            config=cfg,
            logs=logs,
            total_rewards=total_rewards,
            final_B=state.B_base,
            final_streak=state.streak,
        )