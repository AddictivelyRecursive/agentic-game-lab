"""
experiments/run_cross_play_tournament.py

Purpose:
- Run an ordered round-robin cross-play tournament across a mixed roster of
  OpenRouter-backed LLM agents and deterministic baseline agents.
- Write logs in a path structure that is easy to analyze later.

Expected output structure:
results/
  cross_play/
    <experiment_id>/
      manifest.json
      <match_id>/
        match_manifest.json
        episode_meta.json
        episode_logs.jsonl
        agents/
          p0__<row_player_slug>/   # only populated for LLM agents
          p1__<col_player_slug>/   # only populated for LLM agents
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    # Fine if python-dotenv is not installed; env vars can still come from shell.
    pass

from game_engine.agents import (
    AlwaysCooperate,
    AlwaysDefect,
    GradedTFT,
    LLMWrapperAgent,
)
from game_engine.env.simulator import GameSimulator
from game_engine.env.types import (
    DriftConfig,
    EnvConfig,
    ObservationConfig,
    PayoffConfig,
    StreakConfig,
)
from game_engine.io.jsonl import write_episode
from game_engine.io.run_paths import (
    build_experiment_id,
    build_match_id,
    ensure_dir,
    slugify,
    write_json,
)


@dataclass(frozen=True)
class PlayerSpec:
    kind: str  # "llm" or "deterministic"
    label: str

    # LLM fields
    model_name: Optional[str] = None
    backend: str = "openrouter"

    # deterministic fields
    strategy: Optional[str] = None
    kwargs: Dict[str, Any] = field(default_factory=dict)


PROMPT_DIR = "AI_Agent/prompts"
RESULTS_ROOT = os.path.join("results", "cross_play")

# Controlled cross-play setup:
# - N=2
# - no perception noise
# - no streak incentive
# - no drift
# This isolates player-vs-player behavior first.
BASE_CFG = EnvConfig(
    N=2,
    M=2,
    T=50,
    p_perception=0.00,
    payoff=PayoffConfig(
        B0=2.0,
        C=1.0,
        K=0.0,
        B_min=2.0,
        B_max=2.0,
    ),
    drift=DriftConfig(
        window_w=10,
        eta=0.00,
        r_star=0.5,
    ),
    streak=StreakConfig(
        theta=0.6,
        lam=0.00,
        tau=5.0,
    ),
    obs=ObservationConfig(
        history_k=10,
        stats_window=10,
    ),
    seed=0,
)

SEEDS: List[int] = [101]
INCLUDE_SELF_PLAY = True

# 10 total players:
# - 7 OpenRouter LLMs
# - 3 deterministic strategies
#
# If any OpenRouter model ID is unavailable for your account/provider routing,
# replace just that model_name string and keep the same label.
PLAYER_SPECS: List[PlayerSpec] = [
    PlayerSpec(
        kind="llm",
        label="gpt4o_mini",
        model_name="openai/gpt-4o-mini",
    ),
    PlayerSpec(
        kind="llm",
        label="claude_haiku",
        model_name="anthropic/claude-3.5-haiku",
    ),
    PlayerSpec(
        kind="llm",
        label="gemini_25_pro",
        model_name="google/gemini-2.5-pro",
    ),
    PlayerSpec(
        kind="llm",
        label="deepseek_v3_0324",
        model_name="deepseek/deepseek-chat-v3-0324",
    ),
    PlayerSpec(
        kind="llm",
        label="qwen25_72b",
        model_name="qwen/qwen-2.5-72b-instruct",
    ),
    PlayerSpec(
        kind="llm",
        label="llama33_70b",
        model_name="meta-llama/llama-3.3-70b-instruct",
    ),
    PlayerSpec(
        kind="llm",
        label="grok_41_fast",
        model_name="x-ai/grok-4.1-fast",
    ),
    PlayerSpec(
        kind="deterministic",
        label="always_cooperate",
        strategy="always_cooperate",
    ),
    PlayerSpec(
        kind="deterministic",
        label="always_defect",
        strategy="always_defect",
    ),
    PlayerSpec(
        kind="deterministic",
        label="graded_tft",
        strategy="graded_tft",
    ),
]


def _has_any_llm(specs: Sequence[PlayerSpec]) -> bool:
    return any(spec.kind == "llm" for spec in specs)


def _require_openrouter_key_if_needed(specs: Sequence[PlayerSpec]) -> None:
    if not _has_any_llm(specs):
        return

    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise EnvironmentError(
            "OPENROUTER_API_KEY not found. Put it in .env or export it in your shell."
        )


def _spec_label(spec: PlayerSpec) -> str:
    return slugify(spec.label, max_len=64)


def _iter_pairs(
    specs: Sequence[PlayerSpec],
    include_self_play: bool,
) -> List[Tuple[PlayerSpec, PlayerSpec]]:
    pairs: List[Tuple[PlayerSpec, PlayerSpec]] = []
    for row in specs:
        for col in specs:
            if not include_self_play and _spec_label(row) == _spec_label(col):
                continue
            pairs.append((row, col))
    return pairs


def _build_agent(
    *,
    spec: PlayerSpec,
    seat: int,
    cfg: EnvConfig,
    match_agents_dir: str,
):
    label = _spec_label(spec)

    if spec.kind == "llm":
        seat_dir = ensure_dir(os.path.join(match_agents_dir, f"p{seat}__{label}"))
        return LLMWrapperAgent(
            name=f"p{seat}__{label}",
            agent_id=seat,
            env_cfg=cfg,
            backend=spec.backend,
            model_name=spec.model_name or "",
            prompt_dir=PROMPT_DIR,
            output_dir=seat_dir,
        )

    if spec.kind == "deterministic":
        if spec.strategy == "always_cooperate":
            return AlwaysCooperate(name=f"p{seat}__{label}")
        if spec.strategy == "always_defect":
            return AlwaysDefect(name=f"p{seat}__{label}")
        if spec.strategy == "graded_tft":
            return GradedTFT(name=f"p{seat}__{label}")
        raise ValueError(f"Unknown deterministic strategy: {spec.strategy!r}")

    raise ValueError(f"Unknown player kind: {spec.kind!r}")


def _player_manifest_obj(spec: PlayerSpec, seat: int) -> Dict[str, Any]:
    return {
        "label": _spec_label(spec),
        "kind": spec.kind,
        "model_name": spec.model_name if spec.kind == "llm" else None,
        "backend": spec.backend if spec.kind == "llm" else None,
        "strategy": spec.strategy if spec.kind == "deterministic" else None,
        "seat": seat,
    }


def main() -> None:
    _require_openrouter_key_if_needed(PLAYER_SPECS)

    assert BASE_CFG.N == 2, "Cross-play tournament runner expects N=2."
    assert len(PLAYER_SPECS) >= 2, "Need at least 2 players for cross-play."
    assert (
        BASE_CFG.payoff.B_min * (1.0 + BASE_CFG.streak.lam) > BASE_CFG.payoff.C
    ), "PD constraint violated: B_eff may drop <= C"

    labels = [_spec_label(spec) for spec in PLAYER_SPECS]
    assert len(labels) == len(set(labels)), "Player labels must be unique."

    pair_list = _iter_pairs(PLAYER_SPECS, INCLUDE_SELF_PLAY)

    # Assumes your updated run_paths.build_experiment_id() signature is:
    # build_experiment_id(prefix, cfg=..., num_seeds=..., extra_tags=...)
    run_id = build_experiment_id(
        "cp",
        cfg=BASE_CFG,
        num_seeds=len(SEEDS),
        extra_tags=[f"K{len(PLAYER_SPECS)}"],
    )
    run_dir = ensure_dir(os.path.join(RESULTS_ROOT, run_id))

    manifest = {
        "run_id": run_id,
        "experiment_type": "cross_play",
        "prompt_dir": PROMPT_DIR,
        "include_self_play": INCLUDE_SELF_PLAY,
        "num_players": len(PLAYER_SPECS),
        "num_seeds": len(SEEDS),
        "num_pairs": len(pair_list),
        "num_matches_expected": len(pair_list) * len(SEEDS),
        "players": [
            {
                "label": _spec_label(spec),
                "kind": spec.kind,
                "model_name": spec.model_name if spec.kind == "llm" else None,
                "backend": spec.backend if spec.kind == "llm" else None,
                "strategy": spec.strategy if spec.kind == "deterministic" else None,
            }
            for spec in PLAYER_SPECS
        ],
        "config": BASE_CFG,
    }
    write_json(os.path.join(run_dir, "manifest.json"), manifest)

    completed = 0
    total = len(pair_list) * len(SEEDS)

    for row_spec, col_spec in pair_list:
        row_label = _spec_label(row_spec)
        col_label = _spec_label(col_spec)

        for seed in SEEDS:
            cfg = replace(BASE_CFG, seed=seed)

            match_id = build_match_id(row_label, col_label, seed=seed)
            match_dir = ensure_dir(os.path.join(run_dir, match_id))
            agents_dir = ensure_dir(os.path.join(match_dir, "agents"))

            match_manifest = {
                "run_id": run_id,
                "match_id": match_id,
                "seed": seed,
                "row_player": _player_manifest_obj(row_spec, seat=0),
                "col_player": _player_manifest_obj(col_spec, seat=1),
                "config": cfg,
            }
            write_json(os.path.join(match_dir, "match_manifest.json"), match_manifest)

            try:
                agents = [
                    _build_agent(
                        spec=row_spec,
                        seat=0,
                        cfg=cfg,
                        match_agents_dir=agents_dir,
                    ),
                    _build_agent(
                        spec=col_spec,
                        seat=1,
                        cfg=cfg,
                        match_agents_dir=agents_dir,
                    ),
                ]

                sim = GameSimulator(cfg)
                result = sim.run_episode(agents)

                extra_meta: Dict[str, Any] = {
                    "match_id": match_id,
                    "seed": seed,
                    "row_player_label": row_label,
                    "row_player_kind": row_spec.kind,
                    "row_player_model_id": row_spec.model_name
                    if row_spec.kind == "llm"
                    else None,
                    "row_player_strategy": row_spec.strategy
                    if row_spec.kind == "deterministic"
                    else None,
                    "col_player_label": col_label,
                    "col_player_kind": col_spec.kind,
                    "col_player_model_id": col_spec.model_name
                    if col_spec.kind == "llm"
                    else None,
                    "col_player_strategy": col_spec.strategy
                    if col_spec.kind == "deterministic"
                    else None,
                    "seat_assignment": {
                        "0": row_label,
                        "1": col_label,
                    },
                }

                meta_path, logs_path = write_episode(
                    match_dir,
                    run_id,
                    0,
                    result,
                    extra_meta=extra_meta,
                    stem="episode",
                )

                completed += 1
                print(
                    f"[{completed}/{total}] "
                    f"{row_label} vs {col_label} | seed={seed} | "
                    f"meta={meta_path} | logs={logs_path}"
                )

            except Exception as e:
                write_json(
                    os.path.join(match_dir, "error.json"),
                    {
                        "run_id": run_id,
                        "match_id": match_id,
                        "seed": seed,
                        "row_player": _player_manifest_obj(row_spec, seat=0),
                        "col_player": _player_manifest_obj(col_spec, seat=1),
                        "error": repr(e),
                    },
                )
                print(f"[ERROR] {match_id}: {e}")

    print("\nDone.")
    print("Run dir:", run_dir)


if __name__ == "__main__":
    main()