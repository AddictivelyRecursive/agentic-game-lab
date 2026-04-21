"""
experiments/run_llmvsbaseline.py

Purpose:
- Paper-style LLM-vs-baseline evaluation for 2-player binary IPD.
- Each focal LLM plays against a fixed suite of baseline strategies:
  AlwaysCooperate, AlwaysDefect, Random, GradedTFT, GrimTrigger, WSLS.
- Writes logs with the same general structure as run_crossplay_tournament.py
  and run_hostility_sweep.py.

Output structure:
results/
  llm_vs_baseline/
    <run_id>/
      manifest.json
      <match_id>/
        match_manifest.json
        episode_meta.json
        episode_logs.jsonl
        agents/
          p0__<llm_label>/   # when LLM is in seat 0
          p1__<llm_label>/   # only if SWAP_SEATS=True and LLM is in seat 1
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
    RandomAgent,
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
    kind: str  # "llm" or "baseline"
    label: str

    # LLM fields
    model_name: Optional[str] = None
    backend: str = "openrouter"

    # Baseline fields
    strategy: Optional[str] = None
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MatchSpec:
    focal_spec: PlayerSpec
    baseline_spec: PlayerSpec
    row_spec: PlayerSpec
    col_spec: PlayerSpec
    matchup_type: str  # "focal_vs_baseline"
    seat_swapped: bool


PROMPT_DIR = "AI_Agent/prompts"
RESULTS_ROOT = os.path.join("results", "llm_vs_baseline")

# Controlled binary IPD setup
BASE_CFG = EnvConfig(
    N=2,
    M=2,
    T=100,
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
    obs=ObservationConfig(history_k=10, stats_window=10),
    seed=0,
)

# One seed for now -> 5 focal models x 6 baselines = 30 matches.
# If SWAP_SEATS=True, that doubles to 60 matches.
SEEDS: List[int] = [101]

# Keep False for cheaper runs. Flip to True if you want seat-position control.
SWAP_SEATS = False

FOCAL_PLAYER_SPECS: List[PlayerSpec] = [
    PlayerSpec(
        kind="llm",
        label="deepseek_v32",
        model_name="deepseek/deepseek-v3.2",
    ),
    PlayerSpec(
        kind="llm",
        label="qwen3_235b_a22b_2507",
        model_name="qwen/qwen3-235b-a22b-2507",
    ),
    PlayerSpec(
        kind="llm",
        label="gpt_oss_20b",
        model_name="openai/gpt-oss-20b",
    ),
    PlayerSpec(
        kind="llm",
        label="gemma3_27b",
        model_name="google/gemma-3-27b-it",
    ),
    PlayerSpec(
        kind="llm",
        label="llama31_8b",
        model_name="meta-llama/llama-3.1-8b-instruct",
    ),
]

BASELINE_SPECS: List[PlayerSpec] = [
    PlayerSpec(
        kind="baseline",
        label="always_cooperate",
        strategy="always_cooperate",
    ),
    PlayerSpec(
        kind="baseline",
        label="always_defect",
        strategy="always_defect",
    ),
    PlayerSpec(
        kind="baseline",
        label="random_uniform",
        strategy="random_uniform",
    ),
    PlayerSpec(
        kind="baseline",
        label="graded_tft",
        strategy="graded_tft",
    ),
    PlayerSpec(
        kind="baseline",
        label="grim_trigger",
        strategy="grim_trigger",
    ),
    PlayerSpec(
        kind="baseline",
        label="wsls",
        strategy="wsls",
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


def _iter_matchups(
    focals: Sequence[PlayerSpec],
    baselines: Sequence[PlayerSpec],
    *,
    swap_seats: bool,
) -> List[MatchSpec]:
    jobs: List[MatchSpec] = []

    for focal in focals:
        for baseline in baselines:
            # Default: LLM in seat 0, baseline in seat 1
            jobs.append(
                MatchSpec(
                    focal_spec=focal,
                    baseline_spec=baseline,
                    row_spec=focal,
                    col_spec=baseline,
                    matchup_type="focal_vs_baseline",
                    seat_swapped=False,
                )
            )

            # Optional control: baseline in seat 0, LLM in seat 1
            if swap_seats:
                jobs.append(
                    MatchSpec(
                        focal_spec=focal,
                        baseline_spec=baseline,
                        row_spec=baseline,
                        col_spec=focal,
                        matchup_type="focal_vs_baseline",
                        seat_swapped=True,
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

    if spec.kind == "baseline":
        strategy = spec.strategy

        if strategy == "always_cooperate":
            return AlwaysCooperate(name=f"p{seat}__{label}")

        if strategy == "always_defect":
            return AlwaysDefect(name=f"p{seat}__{label}")

        if strategy == "random_uniform":
            return RandomAgent(name=f"p{seat}__{label}")

        if strategy == "graded_tft":
            return GradedTFT(name=f"p{seat}__{label}")

        if strategy == "grim_trigger":
            return GrimTrigger(name=f"p{seat}__{label}", **spec.kwargs)

        if strategy == "wsls":
            return WinStayLoseShift(name=f"p{seat}__{label}", **spec.kwargs)

        raise ValueError(f"Unknown baseline strategy: {strategy!r}")

    raise ValueError(f"Unknown player kind: {spec.kind!r}")


def _player_manifest_obj(spec: PlayerSpec, seat: int) -> Dict[str, Any]:
    return {
        "label": _spec_label(spec),
        "kind": spec.kind,
        "model_name": spec.model_name if spec.kind == "llm" else None,
        "backend": spec.backend if spec.kind == "llm" else None,
        "strategy": spec.strategy if spec.kind == "baseline" else None,
        "kwargs": spec.kwargs if spec.kind == "baseline" else None,
        "seat": seat,
    }


def main() -> None:
    _require_openrouter_key_if_needed(FOCAL_PLAYER_SPECS)

    assert BASE_CFG.N == 2, "This runner is defined for 2-player games only."
    assert BASE_CFG.M == 2, "This runner is defined for binary-action games only."
    assert (
        BASE_CFG.payoff.B_min * (1.0 + BASE_CFG.streak.lam) > BASE_CFG.payoff.C
    ), "PD constraint violated: B_eff may drop <= C"

    all_specs = list(FOCAL_PLAYER_SPECS) + list(BASELINE_SPECS)
    labels = [_spec_label(spec) for spec in all_specs]
    assert len(labels) == len(set(labels)), "Player labels must be unique."

    match_list = _iter_matchups(
        FOCAL_PLAYER_SPECS,
        BASELINE_SPECS,
        swap_seats=SWAP_SEATS,
    )

    run_id = build_experiment_id(
        "lvb",
        cfg=BASE_CFG,
        num_seeds=len(SEEDS),
        extra_tags=[
            f"F{len(FOCAL_PLAYER_SPECS)}",
            f"B{len(BASELINE_SPECS)}",
            f"swap{int(SWAP_SEATS)}",
            "T100",
        ],
    )
    run_dir = ensure_dir(os.path.join(RESULTS_ROOT, run_id))

    manifest = {
        "run_id": run_id,
        "experiment_type": "llm_vs_baseline",
        "prompt_dir": PROMPT_DIR,
        "swap_seats": SWAP_SEATS,
        "num_focals": len(FOCAL_PLAYER_SPECS),
        "num_baselines": len(BASELINE_SPECS),
        "num_seeds": len(SEEDS),
        "num_match_specs": len(match_list),
        "num_matches_expected": len(match_list) * len(SEEDS),
        "focal_players": [
            {
                "label": _spec_label(spec),
                "kind": spec.kind,
                "model_name": spec.model_name,
                "backend": spec.backend,
            }
            for spec in FOCAL_PLAYER_SPECS
        ],
        "baseline_players": [
            {
                "label": _spec_label(spec),
                "kind": spec.kind,
                "strategy": spec.strategy,
                "kwargs": spec.kwargs,
            }
            for spec in BASELINE_SPECS
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
        baseline_spec = match.baseline_spec

        row_label = _spec_label(row_spec)
        col_label = _spec_label(col_spec)
        focal_label = _spec_label(focal_spec)
        baseline_label = _spec_label(baseline_spec)

        focal_seat = 1 if match.seat_swapped else 0
        baseline_seat = 0 if match.seat_swapped else 1

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
                "seat_swapped": match.seat_swapped,
                "focal_seat": focal_seat,
                "baseline_seat": baseline_seat,
                "focal_player": {
                    "label": focal_label,
                    "kind": focal_spec.kind,
                    "model_name": focal_spec.model_name,
                    "backend": focal_spec.backend,
                },
                "baseline_player": {
                    "label": baseline_label,
                    "kind": baseline_spec.kind,
                    "strategy": baseline_spec.strategy,
                    "kwargs": baseline_spec.kwargs,
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
                    "seat_swapped": match.seat_swapped,
                    "focal_seat": focal_seat,
                    "baseline_seat": baseline_seat,
                    "focal_player_label": focal_label,
                    "focal_player_kind": focal_spec.kind,
                    "focal_player_model_id": focal_spec.model_name,
                    "baseline_player_label": baseline_label,
                    "baseline_player_kind": baseline_spec.kind,
                    "baseline_player_strategy": baseline_spec.strategy,
                    "row_player_label": row_label,
                    "row_player_kind": row_spec.kind,
                    "row_player_model_id": (
                        row_spec.model_name if row_spec.kind == "llm" else None
                    ),
                    "row_player_strategy": (
                        row_spec.strategy if row_spec.kind == "baseline" else None
                    ),
                    "col_player_label": col_label,
                    "col_player_kind": col_spec.kind,
                    "col_player_model_id": (
                        col_spec.model_name if col_spec.kind == "llm" else None
                    ),
                    "col_player_strategy": (
                        col_spec.strategy if col_spec.kind == "baseline" else None
                    ),
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
                    f"type={match.matchup_type} | "
                    f"swapped={int(match.seat_swapped)} | "
                    f"seed={seed} | "
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
                        "seat_swapped": match.seat_swapped,
                        "focal_player": {
                            "label": focal_label,
                            "kind": focal_spec.kind,
                            "model_name": focal_spec.model_name,
                            "backend": focal_spec.backend,
                        },
                        "baseline_player": {
                            "label": baseline_label,
                            "kind": baseline_spec.kind,
                            "strategy": baseline_spec.strategy,
                            "kwargs": baseline_spec.kwargs,
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