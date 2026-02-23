"""
env/types.py

Purpose:
- Defines pure dataclasses / schemas used across env, agents, and evaluation.
- No computation should live here.

Return values:
- These types are used as structured inputs/outputs for simulation and logging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class PayoffConfig:
    """Parameters for payoff computation.

    B0: initial base benefit parameter B_t.
    C: cost coefficient for own cooperation.
    K: optional additive offset (often 0).
    B_min/B_max: hard bounds applied to B_t under drift. Must ensure PD constraints.
    """
    B0: float
    C: float
    K: float = 0.0
    B_min: float = 0.0
    B_max: float = 10.0


@dataclass(frozen=True)
class DriftConfig:
    """q3-global drift parameters.

    window_w: rolling window length for observed global cooperation mean.
    eta: drift rate.
    r_star: reference cooperation (neutral point).
    """
    window_w: int
    eta: float
    r_star: float


@dataclass(frozen=True)
class StreakConfig:
    """Global streak parameters (based on observed global cooperation mean).

    theta: threshold on observed global mean cooperation to increment streak.
    lam: max multiplier strength for B_eff via streak bonus.
    tau: saturation factor in tanh(streak/tau).
    """
    theta: float
    lam: float
    tau: float


@dataclass(frozen=True)
class ObservationConfig:
    """Observation configuration.

    history_k: number of previous timesteps to include in observed history.
    stats_window: window length for per-player cm (mean observed cooperation); if None, uses min(t-1, drift.window_w).
    """
    history_k: int = 10
    stats_window: Optional[int] = None


@dataclass(frozen=True)
class EnvConfig:
    """Environment config (fully determines the environment dynamics given a seed)."""
    N: int
    M: int
    T: int
    p_perception: float
    payoff: PayoffConfig
    drift: DriftConfig
    streak: StreakConfig
    obs: ObservationConfig = field(default_factory=ObservationConfig)
    seed: int = 0


@dataclass(frozen=True)
class AgentMeta:
    """Per-step metadata emitted by an agent (useful for LLMs, but generic)."""
    agent_name: str
    parse_ok: bool = True
    action_ok: bool = True
    fallback_used: bool = False
    fallback_reason: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    raw_hash: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Observation:
    """What the environment provides to an agent at round t (built only from history up to t-1)."""
    t: int
    agent_id: int
    N: int
    M: int
    p: float

    # Current payoff parameters for round t (computed using streak up to t-1).
    B_base: float
    B_eff: float
    C: float
    K: float

    # q3-global public signals up to t-1
    r_obs_prev: float
    streak_prev: int

    # Rules (constants; include so stateless agents can reason)
    drift: DriftConfig
    streak: StreakConfig

    # Observed history up to t-1 (packed or raw; packed helps token budget)
    # Each string is length <= k, digits are actions 0..M-1 (works for M<=10).
    observed_history_packed: List[str]

    # Cheap per-player summary over a window (mean observed cooperation)
    coop_mean: List[float]


@dataclass(frozen=True)
class RoundLog:
    """Authoritative per-round record (one per timestep)."""
    t: int
    N: int
    M: int

    # Parameters used this round (before updates)
    B_base: float
    B_eff: float
    C: float
    K: float
    p: float

    r_obs_prev: float
    streak_prev: int

    # Actions and cooperation levels
    true_actions: List[int]
    obs_actions: List[int]
    true_coop: List[float]
    obs_coop: List[float]
    obs_coop_mean: float

    # Payoffs (computed from true actions)
    rewards: List[float]

    # Updates computed after this round
    r_obs_next: float
    streak_next: int
    B_next: float

    agent_meta: List[AgentMeta]


@dataclass(frozen=True)
class EpisodeResult:
    """Episode output (in-memory). Write to JSONL outside env."""
    config: EnvConfig
    logs: List[RoundLog]
    total_rewards: List[float]
    final_B: float
    final_streak: int