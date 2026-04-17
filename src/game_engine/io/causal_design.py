from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

from game_engine.env.types import EnvConfig
from game_engine.io.run_paths import slugify


@dataclass(frozen=True)
class PlayerSpec:
    kind: str  # "llm" or "deterministic"
    label: str

    # LLM fields
    model_name: str | None = None
    backend: str = "openrouter"

    # deterministic fields
    strategy: str | None = None
    kwargs: Dict[str, Any] = field(default_factory=dict)


STANDARD_PLAYER_REGISTRY: Dict[str, PlayerSpec] = {
    "always_cooperate": PlayerSpec(
        kind="deterministic",
        label="always_cooperate",
        strategy="always_cooperate",
    ),
    "always_defect": PlayerSpec(
        kind="deterministic",
        label="always_defect",
        strategy="always_defect",
    ),
    "graded_tft": PlayerSpec(
        kind="deterministic",
        label="graded_tft",
        strategy="graded_tft",
    ),
}

LLM_PLAYER_REGISTRY: Dict[str, PlayerSpec] = {
    "gpt4o_mini": PlayerSpec(
        kind="llm",
        label="gpt4o_mini",
        model_name="openai/gpt-4o-mini",
    ),
    "claude_haiku": PlayerSpec(
        kind="llm",
        label="claude_haiku",
        model_name="anthropic/claude-3.5-haiku",
    ),
    "gemini_25_pro": PlayerSpec(
        kind="llm",
        label="gemini_25_pro",
        model_name="google/gemini-2.5-pro",
    ),
    "deepseek_v3_0324": PlayerSpec(
        kind="llm",
        label="deepseek_v3_0324",
        model_name="deepseek/deepseek-chat-v3-0324",
    ),
    "qwen25_72b": PlayerSpec(
        kind="llm",
        label="qwen25_72b",
        model_name="qwen/qwen-2.5-72b-instruct",
    ),
    "llama33_70b": PlayerSpec(
        kind="llm",
        label="llama33_70b",
        model_name="meta-llama/llama-3.3-70b-instruct",
    ),
    "grok_41_fast": PlayerSpec(
        kind="llm",
        label="grok_41_fast",
        model_name="x-ai/grok-4.1-fast",
    ),
}

ALIASES: Dict[str, str] = {
    "grated_tft": "graded_tft",
}


def canonicalize_name(name: str) -> str:
    key = slugify(name.strip(), max_len=64)
    return ALIASES.get(key, key)


def safe_label(spec: PlayerSpec) -> str:
    return slugify(spec.label, max_len=64)


def resolve_standard_players(names: Sequence[str]) -> List[PlayerSpec]:
    out: List[PlayerSpec] = []
    unknown: List[str] = []

    for raw_name in names:
        name = canonicalize_name(raw_name)
        spec = STANDARD_PLAYER_REGISTRY.get(name)
        if spec is None:
            unknown.append(raw_name)
        else:
            out.append(spec)

    if unknown:
        raise ValueError(
            f"Unknown standard strategies: {unknown}. "
            f"Valid options: {sorted(STANDARD_PLAYER_REGISTRY.keys())}"
        )

    return out


def resolve_llm_players(names: Sequence[str]) -> List[PlayerSpec]:
    out: List[PlayerSpec] = []
    unknown: List[str] = []

    for raw_name in names:
        name = canonicalize_name(raw_name)
        spec = LLM_PLAYER_REGISTRY.get(name)
        if spec is None:
            unknown.append(raw_name)
        else:
            out.append(spec)

    if unknown:
        raise ValueError(
            f"Unknown LLM names: {unknown}. "
            f"Valid options: {sorted(LLM_PLAYER_REGISTRY.keys())}"
        )

    return out


def build_lineup(
    standard_names: Sequence[str],
    llm_names: Sequence[str],
) -> List[PlayerSpec]:
    standard_specs = resolve_standard_players(standard_names)
    llm_specs = resolve_llm_players(llm_names)

    lineup = standard_specs + llm_specs
    if len(lineup) < 2:
        raise ValueError(
            "Need at least 2 total players. "
            "Make sure len(standard) + len(llms) >= 2."
        )

    labels = [safe_label(spec) for spec in lineup]
    if len(labels) != len(set(labels)):
        raise ValueError(
            "Duplicate player names are not allowed in this simple lineup-driven engine. "
            "Use each strategy/model at most once per run."
        )

    return lineup


def lineup_label(lineup: Sequence[PlayerSpec]) -> str:
    return "__".join(safe_label(spec) for spec in lineup)


def player_manifest_obj(spec: PlayerSpec, seat: int) -> Dict[str, Any]:
    return {
        "label": safe_label(spec),
        "kind": spec.kind,
        "model_name": spec.model_name if spec.kind == "llm" else None,
        "backend": spec.backend if spec.kind == "llm" else None,
        "strategy": spec.strategy if spec.kind == "deterministic" else None,
        "seat": int(seat),
    }


def validate_social_dilemma_cfg(cfg: EnvConfig) -> None:
    if cfg.N < 2:
        raise ValueError("Need N >= 2.")
    if cfg.M < 2:
        raise ValueError("Need M >= 2.")
    if cfg.M > 10:
        raise ValueError(
            "Current packed-history format assumes M <= 10."
        )
    if cfg.T != 50:
        raise ValueError("This study keeps T fixed at 50.")
    if not (0.0 <= cfg.p_perception <= 1.0):
        raise ValueError("p_perception must be in [0, 1].")
    if cfg.payoff.C <= 0.0:
        raise ValueError("Need C > 0 for a strict social dilemma.")
    if cfg.payoff.B_min <= cfg.payoff.C:
        raise ValueError(
            "Need B_min > C so every reachable round remains in the same dilemma regime."
        )
    if cfg.payoff.B_max < cfg.payoff.B_min:
        raise ValueError("Need B_max >= B_min.")
    if not (cfg.payoff.B_min <= cfg.payoff.B0 <= cfg.payoff.B_max):
        raise ValueError("Need B0 in [B_min, B_max].")
    if cfg.streak.lam < 0.0:
        raise ValueError("This runner assumes streak lambda >= 0.")
    if cfg.obs.history_k < 1:
        raise ValueError("history_k must be >= 1.")
    if cfg.obs.stats_window is not None and cfg.obs.stats_window < 1:
        raise ValueError("stats_window must be >= 1 when provided.")