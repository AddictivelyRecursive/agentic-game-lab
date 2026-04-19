"""
experiments/run_cross_play_tournament.py

Purpose:
- Run a focal-vs-anchor tournament instead of a full round-robin.
- Evaluate each focal LLM against a fixed set of deterministic anchor baselines.
- Each focal LLM plays each deterministic anchor exactly once.

Expected output structure:
results/
  anchor_baseline/
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
from typing import Any, Dict, List, Optional, Sequence

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
    GrimTrigger,
    LLMWrapperAgent,
    WinStayLoseShift,
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


@dataclass(frozen=True)
class MatchSpec:
    row_spec: PlayerSpec
    col_spec: PlayerSpec
    focal_spec: PlayerSpec
    anchor_spec: PlayerSpec
    matchup_type: str  # always "focal_vs_anchor" in this runner


PROMPT_DIR = "AI_Agent/prompts"
RESULTS_ROOT = os.path.join("results", "anchor_baseline")

# Controlled setup:
# - N=2
# - no perception noise
# - no streak incentive
# - no drift
# This isolates strategic behavior first.
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

SWAP_SEATS = False
INCLUDE_FOCAL_SELF_PLAY = False

# Cheaper focal models only.
FOCAL_PLAYER_SPECS: List[PlayerSpec] = [
    PlayerSpec(
        kind="llm",
        label="gpt4o_mini",
        model_name="openai/gpt-4o-mini",
    ),
    PlayerSpec(
        kind="llm",
        label="claude_haiku_45",
        model_name="anthropic/claude-haiku-4.5",
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
]

# 5 deterministic anchors only.
ANCHOR_SPECS: List[PlayerSpec] = [
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


def _iter_anchor_matchups(
    focals: Sequence[PlayerSpec],
    anchors: Sequence[PlayerSpec],
) -> List[MatchSpec]:
    jobs: List[MatchSpec] = []

    for focal in focals:
        for anchor in anchors:
            jobs.append(
                MatchSpec(
                    row_spec=focal,
                    col_spec=anchor,
                    focal_spec=focal,
                    anchor_spec=anchor,
                    matchup_type="focal_vs_anchor",
                )
            )

    return jobs


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
        if spec.strategy == "grim_trigger":
            return GrimTrigger(name=f"p{seat}__{label}")
        if spec.strategy == "wsls":
            return WinStayLoseShift(name=f"p{seat}__{label}")
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
    _require_openrouter_key_if_needed(FOCAL_PLAYER_SPECS)

    assert SEEDS == [101], "This configuration expects exactly one seed: 101."
    assert (
        BASE_CFG.payoff.B_min * (1.0 + BASE_CFG.streak.lam) > BASE_CFG.payoff.C
    ), "PD constraint violated: B_eff may drop <= C"

    all_specs = list(FOCAL_PLAYER_SPECS) + list(ANCHOR_SPECS)
    labels = [_spec_label(spec) for spec in all_specs]
    assert len(labels) == len(set(labels)), "Player labels must be unique across focals and anchors."

    match_list = _iter_anchor_matchups(
        FOCAL_PLAYER_SPECS,
        ANCHOR_SPECS,
    )

    run_id = build_experiment_id(
        "ab",
        cfg=BASE_CFG,
        num_seeds=len(SEEDS),
        extra_tags=[
            f"F{len(FOCAL_PLAYER_SPECS)}",
            f"A{len(ANCHOR_SPECS)}",
            "swap0",
            "self0",
        ],
    )
    run_dir = ensure_dir(os.path.join(RESULTS_ROOT, run_id))

    manifest = {
        "run_id": run_id,
        "experiment_type": "anchor_baseline",
        "prompt_dir": PROMPT_DIR,
        "swap_seats": SWAP_SEATS,
        "include_focal_self_play": INCLUDE_FOCAL_SELF_PLAY,
        "num_focals": len(FOCAL_PLAYER_SPECS),
        "num_anchors": len(ANCHOR_SPECS),
        "num_seeds": len(SEEDS),
        "num_match_specs": len(match_list),
        "num_matches_expected": len(match_list) * len(SEEDS),
        "focal_players": [
            {
                "label": _spec_label(spec),
                "kind": spec.kind,
                "model_name": spec.model_name if spec.kind == "llm" else None,
                "backend": spec.backend if spec.kind == "llm" else None,
                "strategy": spec.strategy if spec.kind == "deterministic" else None,
            }
            for spec in FOCAL_PLAYER_SPECS
        ],
        "anchor_players": [
            {
                "label": _spec_label(spec),
                "kind": spec.kind,
                "model_name": spec.model_name if spec.kind == "llm" else None,
                "backend": spec.backend if spec.kind == "llm" else None,
                "strategy": spec.strategy if spec.kind == "deterministic" else None,
            }
            for spec in ANCHOR_SPECS
        ],
        "config": BASE_CFG,
    }
    write_json(os.path.join(run_dir, "manifest.json"), manifest)

    completed = 0
    total = len(match_list) * len(SEEDS)

    for match in match_list:
        row_spec = match.row_spec
        col_spec = match.col_spec
        focal_spec = match.focal_spec
        anchor_spec = match.anchor_spec

        row_label = _spec_label(row_spec)
        col_label = _spec_label(col_spec)
        focal_label = _spec_label(focal_spec)
        anchor_label = _spec_label(anchor_spec)

        for seed in SEEDS:
            cfg = replace(BASE_CFG, seed=seed)

            match_id = build_match_id(row_label, col_label, seed=seed)
            match_dir = ensure_dir(os.path.join(run_dir, match_id))
            agents_dir = ensure_dir(os.path.join(match_dir, "agents"))

            match_manifest = {
                "run_id": run_id,
                "match_id": match_id,
                "seed": seed,
                "matchup_type": match.matchup_type,
                "focal_player": {
                    "label": focal_label,
                    "kind": focal_spec.kind,
                    "model_name": focal_spec.model_name if focal_spec.kind == "llm" else None,
                    "backend": focal_spec.backend if focal_spec.kind == "llm" else None,
                    "strategy": focal_spec.strategy if focal_spec.kind == "deterministic" else None,
                },
                "anchor_player": {
                    "label": anchor_label,
                    "kind": anchor_spec.kind,
                    "model_name": anchor_spec.model_name if anchor_spec.kind == "llm" else None,
                    "backend": anchor_spec.backend if anchor_spec.kind == "llm" else None,
                    "strategy": anchor_spec.strategy if anchor_spec.kind == "deterministic" else None,
                },
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
                    "matchup_type": match.matchup_type,
                    "focal_player_label": focal_label,
                    "focal_player_kind": focal_spec.kind,
                    "focal_player_model_id": focal_spec.model_name
                    if focal_spec.kind == "llm"
                    else None,
                    "focal_player_strategy": focal_spec.strategy
                    if focal_spec.kind == "deterministic"
                    else None,
                    "anchor_player_label": anchor_label,
                    "anchor_player_kind": anchor_spec.kind,
                    "anchor_player_model_id": (
                        anchor_spec.model_name if anchor_spec.kind == "llm" else None
                    ),
                    "anchor_player_strategy": (
                        anchor_spec.strategy if anchor_spec.kind == "deterministic" else None
                    ),
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
                    f"{row_label} vs {col_label} | "
                    f"type={match.matchup_type} | seed={seed} | "
                    f"meta={meta_path} | logs={logs_path}"
                )

            except Exception as e:
                write_json(
                    os.path.join(match_dir, "error.json"),
                    {
                        "run_id": run_id,
                        "match_id": match_id,
                        "seed": seed,
                        "matchup_type": match.matchup_type,
                        "focal_player": {
                            "label": focal_label,
                            "kind": focal_spec.kind,
                            "model_name": focal_spec.model_name if focal_spec.kind == "llm" else None,
                            "backend": focal_spec.backend if focal_spec.kind == "llm" else None,
                            "strategy": focal_spec.strategy if focal_spec.kind == "deterministic" else None,
                        },
                        "anchor_player": {
                            "label": anchor_label,
                            "kind": anchor_spec.kind,
                            "model_name": anchor_spec.model_name if anchor_spec.kind == "llm" else None,
                            "backend": anchor_spec.backend if anchor_spec.kind == "llm" else None,
                            "strategy": anchor_spec.strategy if anchor_spec.kind == "deterministic" else None,
                        },
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