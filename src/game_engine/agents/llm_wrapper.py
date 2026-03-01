"""
agents/llm_wrapper.py

Purpose:
- Wrap an external LLM-based agent so it can be plugged into the
  GameSimulator which expects:
    - reset(seed: int) -> None
    - act(obs: Observation) -> (action: int, meta: AgentMeta)

Key points:
- Observation is built only from history up to t-1.
- Observation.observed_history_packed is a List[str] of length N:
    - each string is chronological (oldest -> newest)
    - last character corresponds to action at (t-1)
    - digits encode actions (assumes M <= 10)

Important:
- Your AI_Agent expects observed_history_last_k with "most recent first".
  But simulator provides oldest->newest. This wrapper reverses each row.

Return values:
- act(...) returns a validated action in [0, M-1] and an AgentMeta record.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from AI_Agent.agent.llm_agent import LLMAgent
from AI_Agent.agent.llm_client import OllamaClient
from AI_Agent.agent.dummy_llm_client import DummyLLMClient
from AI_Agent.agent.logger import AgentLogger
from AI_Agent.agent.openrouter_client import OpenRouterClient
from game_engine.env.types import EnvConfig, Observation, AgentMeta


@dataclass(frozen=True)
class ActionSemantics:
    """Mapping from discrete action index to cooperation level in [0,1]."""
    index_to_cooperation: List[float]
    note: str = "Lower index = more cooperation"

    @staticmethod
    def default_for_M(M: int) -> "ActionSemantics":
        # Keep your preferred mapping for M=5 (matches your test-caller)
        if M == 5:
            return ActionSemantics([1.0, 0.75, 0.5, 0.25, 0.0])
        if M < 2:
            return ActionSemantics([1.0])
        step = 1.0 / (M - 1)
        return ActionSemantics([1.0 - i * step for i in range(M)])


def _safe_mkdir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _unpack_history_row(row_packed: str, M: int) -> List[int]:
    """
    Convert packed digit-string row into list[int] actions.
    Returns chronological (oldest -> newest).

    Assumes M <= 10 (single digit per action).
    """
    if not isinstance(row_packed, str) or len(row_packed) == 0:
        return []
    if M > 10:
        raise ValueError("Packed history uses single digits; requires M <= 10.")

    out: List[int] = []
    for ch in row_packed:
        if "0" <= ch <= "9":
            a = int(ch)
            if 0 <= a < M:
                out.append(a)
    return out


def _history_packed_to_matrix(history_packed: List[str], M: int) -> List[List[int]]:
    """Convert List[str] packed history into NxK matrix of ints (ragged per row)."""
    if not isinstance(history_packed, list):
        return []
    return [_unpack_history_row(s, M) for s in history_packed]


def _reverse_rows(history_nxk_oldest_first: List[List[int]]) -> List[List[int]]:
    """Convert NxK oldest->newest rows into newest->oldest rows (index 0 is most recent)."""
    out: List[List[int]] = []
    for row in history_nxk_oldest_first:
        if isinstance(row, list):
            out.append(list(reversed(row)))
        else:
            out.append([])
    return out


def _freq_from_history_matrix(history_nxk: List[List[int]], M: int) -> List[List[float]]:
    """
    Compute NxM empirical action frequencies from an NxK history matrix.
    Order doesn't matter for frequencies.
    """
    if not isinstance(history_nxk, list) or len(history_nxk) == 0:
        return []

    out: List[List[float]] = []
    for row in history_nxk:
        counts = [0] * M
        total = 0
        if isinstance(row, list):
            for a in row:
                if isinstance(a, int) and 0 <= a < M:
                    counts[a] += 1
                    total += 1
        if total == 0:
            out.append([0.0] * M)
        else:
            out.append([c / total for c in counts])
    return out


class LLMWrapperAgent:
    """
    Wraps AI_Agent.LLMAgent for use inside game_engine.env.GameSimulator.

    This wrapper also acts as the "factory" (switch-case) for choosing
    LLM backend/model, enabling model-vs-model experiments.
    """

    def __init__(
        self,
        name: str,
        agent_id: int,
        env_cfg: EnvConfig,
        *,
        backend: str = "ollama",          # "ollama" | "dummy"
        model_name: str = "llama3.1:8b",  # used for ollama backend
        ollama_host: str = "http://localhost:11434",
        dummy_mode: str = "mostly_valid",
        dummy_seed: int = 7,
        dummy_invalid_rate: float = 0.6,
        dummy_force_invalid_first_n6: int = 1,
        prompt_dir: str = "AI_Agent/prompts",
        output_dir: Optional[str] = None,
    ) -> None:
        self.name = name
        self.agent_id = agent_id
        self.env_cfg = env_cfg

        # ---- Switch-case: choose client based on backend ----
        backend_norm = (backend or "").strip().lower()

        if backend_norm == "ollama":
            llm_client = OllamaClient(model_name=model_name, host=ollama_host)

        elif backend_norm == "openrouter":
            # model_name should be an OpenRouter model id, e.g. "openai/gpt-4o-mini"
            llm_client = OpenRouterClient(
                model_name=model_name,
                api_key=os.getenv("OPENROUTER_API_KEY", ""),  # recommended via env
                # Optional: helps OpenRouter analytics; safe to omit
                site_url=os.getenv("OPENROUTER_SITE_URL"),
                app_name=os.getenv("OPENROUTER_APP_NAME", "agentic-game-lab"),
            )

        elif backend_norm == "dummy":
            llm_client = DummyLLMClient(
                mode=dummy_mode,
                seed=dummy_seed,
                invalid_rate=dummy_invalid_rate,
                force_invalid_first_n6=dummy_force_invalid_first_n6,
            )

        else:
            raise ValueError(
                f"Unknown LLM backend: {backend!r}. Supported: 'ollama', 'openrouter', 'dummy'."
            )

        # ---- Logging isolation (recommended) ----
        logger = None
        if output_dir is not None:
            out = _safe_mkdir(output_dir)
            logger = AgentLogger(output_dir=out)

        # ---- Construct external agent with injected client ----
        self.llm = LLMAgent(
            llm_client=llm_client,
            model_name=model_name,          # harmless for dummy; used for fallback default path only
            ollama_host=ollama_host,
            prompt_dir=prompt_dir,
            logger=logger,
            output_dir=None if logger is not None else "AI_Agent/outputs",
        )

        # For meta/debug (not passed to LLM turn payload)
        self._backend = backend_norm
        self._model_name = model_name

    def reset(self, seed: int) -> None:
        """Reset wrapper/LLMAgent state (best-effort)."""
        # Wrapper itself is stateless; LLMAgent currently doesn't use seed.
        if hasattr(self.llm, "reset"):
            try:
                self.llm.reset(seed=seed)
            except TypeError:
                self.llm.reset()

    def act(self, obs: Observation) -> Tuple[int, AgentMeta]:
        """Build turn dict, call LLMAgent.step, validate output."""
        t0 = time.time()

        turn = self._build_turn(obs)

        # External call
        a = self.llm.step(turn)

        # Validate
        action_ok = isinstance(a, int) and 0 <= a < obs.M
        fallback_used = False
        fallback_reason = None

        if not action_ok:
            fallback_used = True
            fallback_reason = "wrapper_invalid_action->fallback_to_0"
            a = 0

        latency_ms = int((time.time() - t0) * 1000)

        # NOTE: Keep identity subtle in turn payload; AgentMeta can include name/model safely.
        meta = AgentMeta(
            agent_name=self.name,
            parse_ok=True,
            action_ok=action_ok,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
            input_tokens=0,
            output_tokens=0,
            latency_ms=latency_ms,
            raw_hash=None,
            extra={
                "agent_type": "LLMWrapperAgent",
                "backend": self._backend,
                "model_name": self._model_name,
            },
        )
        return a, meta

    # ----------------- Mapping: Observation -> exact AI_Agent schema -----------------

    def _build_turn(self, obs: Observation) -> Dict[str, Any]:
        """
        Construct EXACT schema (matching your sample), and keep it clean:
        - no agent names, run ids, seeds, or backend/model info in payload
        - observed_history_last_k is most-recent-first (index 0 is t-1)
        - action_freq_NxM is a plain NxM matrix (list-of-lists)
        """
        cfg = self.env_cfg
        semantics = ActionSemantics.default_for_M(obs.M)

        # Observation.observed_history_packed: List[str], length N
        # Each string is chronological old->new, last char corresponds to (t-1).
        hist_packed: List[str] = list(obs.observed_history_packed)

        # Convert packed strings -> NxK ints, chronological old->new
        hist_oldest_first = _history_packed_to_matrix(hist_packed, obs.M)

        # --- Guard for t=1 (or any case with empty history) ---
        # AI_Agent build_features expects at least one timestep (uses [:,0]).
        # If history is empty, inject a 1-step default action per player.
        if not hist_oldest_first or all(len(row) == 0 for row in hist_oldest_first):
            # Default to "most cooperative" action 0
            hist_oldest_first = [[0] for _ in range(obs.N)]

        # Convert to most-recent-first for LLM input convention
        hist_most_recent_first = _reverse_rows(hist_oldest_first)

        # Compute NxM per-player action frequencies
        action_freq_nxm = _freq_from_history_matrix(hist_oldest_first, obs.M)

        turn: Dict[str, Any] = {
            "round": int(obs.t),
            "agent_id": int(obs.agent_id),

            "game_parameters": {
                "N": int(obs.N),
                "M": int(obs.M),
                "perception_noise_p": float(obs.p),
                "action_semantics": {
                    "index_to_cooperation": list(semantics.index_to_cooperation),
                    "note": semantics.note,
                },
            },

            "payoff": {
                "B_base": float(obs.B_base),
                "B_effective": float(obs.B_eff),
                "C": float(obs.C),
                "K": float(obs.K),
                "formula": "u_i = B_eff * avg_other_coop(true) - C * own_coop(true) + K",
            },

            "streak_rule": {
                "computed_from": "observed_global_mean_coop",
                "theta": float(obs.streak.theta),
                "tau": float(obs.streak.tau),
                "lambda": float(obs.streak.lam),
                "streak_prev": int(obs.streak_prev),
                "streak_update_rule": "if mean_obs_coop >= theta then streak_next=streak_prev+1 else 0",
            },

            "streak_effect_on_payoff": {
                "affects": "B_effective",
                "equation": "B_eff = B_base * (1 + lambda * tanh(streak_prev/tau))",
                "interpretation": "Higher streak_prev increases B_eff",
            },

            "drift_rule": {
                "type": "q3_global",
                "window_w": int(obs.drift.window_w),
                "eta": float(obs.drift.eta),
                "r_star": float(obs.drift.r_star),
                "r_obs_prev": float(obs.r_obs_prev),
                "update_equation": "B_next = clip(B_base + eta*(r_obs_current - r_star), [B_min, B_max])",
                "bounds": {
                    "B_min": float(cfg.payoff.B_min),
                    "B_max": float(cfg.payoff.B_max),
                },
                "note": "r_obs_current computed after this round",
            },

            "information_set": {
                "observed_history_last_k": hist_most_recent_first,
                "rolling_observed_cooperation_prev": float(obs.r_obs_prev),
                "action_freq_NxM": action_freq_nxm,
            },

            "instruction": {
                "task": f"Choose an integer action a in [0,{int(obs.M)-1}] to maximize your long-term average reward.",
                "output_format": "{\"a\": <int>}",
            },
        }

        return turn