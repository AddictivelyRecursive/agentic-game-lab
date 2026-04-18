"""
experiments/run_N_player.py

Purpose:
- Run N-player comparisons.
- N is inferred automatically as len(standard) + len(llms).
- T is fixed to 50.
- We vary:
    * M (action granularity, using the repo's current semantics)
    * perception noise p
    * streak lambda
    * seed
- Logs are written in the same style as cross-play:
    * manifest.json
    * match_manifest.json
    * episode_meta.json
    * episode_logs.jsonl
    * error.json on failure
"""

from __future__ import annotations

import os
from dataclasses import replace
from typing import Any, Dict, List, Sequence

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
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
from game_engine.io.causal_design import (
    PlayerSpec,
    build_lineup,
    lineup_label,
    player_manifest_obj,
    safe_label,
    validate_social_dilemma_cfg,
)
from game_engine.io.jsonl import write_episode
from game_engine.io.run_paths import (
    build_experiment_id,
    ensure_dir,
    slugify,
    write_json,
)

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROMPT_DIR = os.path.join(SRC_ROOT, "AI_Agent", "prompts")
RESULTS_ROOT = os.path.join(SRC_ROOT, "results", "n_player")

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

# Manually edit the lineups for your runs here.
# Either array can be empty, but total players must be >= 2.
STANDARD = []
LLMS = ["gpt4o_mini", "gpt5_mini", "gpt_oss_120b", "gpt54_mini"]
SEEDS = [101]

# Fixed horizon for this study
FIXED_T = 50

# ---------------------------------------------------------------------
# Base config
# ---------------------------------------------------------------------

BASE_CFG = EnvConfig(
    N=2,
    M=2,
    T=FIXED_T,
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

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _require_openrouter_key_if_needed(lineup: Sequence[PlayerSpec]) -> None:
    needs_key = any(
        spec.kind == "llm" and (spec.backend or "").strip().lower() == "openrouter"
        for spec in lineup
    )
    if not needs_key:
        return

    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise EnvironmentError(
            "OPENROUTER_API_KEY not found. Put it in .env or export it in your shell."
        )


def _make_cfg(
    *,
    base_cfg: EnvConfig,
    N: int,
    M: int,
    p: float,
    lam: float,
    seed: int,
) -> EnvConfig:
    cfg = replace(
        base_cfg,
        N=int(N),
        M=int(M),
        T=FIXED_T,
        p_perception=float(p),
        streak=replace(base_cfg.streak, lam=float(lam)),
        seed=int(seed),
    )
    validate_social_dilemma_cfg(cfg)
    return cfg


def _build_agent(
    *,
    spec: PlayerSpec,
    seat: int,
    cfg: EnvConfig,
    match_agents_dir: str,
):
    label = safe_label(spec)

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
            **dict(spec.kwargs or {}),
        )

    if spec.kind == "deterministic":
        strategy = (spec.strategy or "").strip().lower()

        if strategy == "always_cooperate":
            return AlwaysCooperate(name=f"p{seat}__{label}")
        if strategy == "always_defect":
            return AlwaysDefect(name=f"p{seat}__{label}")
        if strategy == "graded_tft":
            return GradedTFT(name=f"p{seat}__{label}")

        raise ValueError(f"Unknown deterministic strategy: {spec.strategy!r}")

    raise ValueError(f"Unknown player kind: {spec.kind!r}")


def _match_id(
    *,
    lineup: Sequence[PlayerSpec],
    N: int,
    M: int,
    p: float,
    lam: float,
    seed: int,
) -> str:
    return slugify(
        f"{lineup_label(lineup)}__N{N}__M{M}__p{p:.2f}__lam{lam:.2f}__seed{seed}",
        max_len=220,
    )


# ---------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------


def run_n_player_engine(
    *,
    standard_names: Sequence[str],
    llm_names: Sequence[str],
    m_values: Sequence[int],
    noise_levels: Sequence[float],
    streak_lambdas: Sequence[float],
    seeds: Sequence[int],
) -> str:
    lineup = build_lineup(standard_names, llm_names)
    _require_openrouter_key_if_needed(lineup)

    N = len(lineup)

    representative_cfg = _make_cfg(
        base_cfg=BASE_CFG,
        N=N,
        M=min(int(x) for x in m_values),
        p=min(float(x) for x in noise_levels),
        lam=min(float(x) for x in streak_lambdas),
        seed=int(seeds[0]),
    )

    run_id = f"{len(llm_names)}__{'__'.join(safe_label(spec) for spec in lineup)}"
    run_dir = ensure_dir(os.path.join(RESULTS_ROOT, run_id))

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "experiment_type": "n_player",
        "prompt_dir": PROMPT_DIR,
        "fixed_T": FIXED_T,
        "selected_standard": list(standard_names),
        "selected_llms": list(llm_names),
        "N": N,
        "M_values": list(m_values),
        "noise_levels": list(noise_levels),
        "streak_lambdas": list(streak_lambdas),
        "seeds": list(seeds),
        "lineup_labels": [safe_label(spec) for spec in lineup],
        "lineup_players": [
            {
                "label": safe_label(spec),
                "kind": spec.kind,
                "model_name": spec.model_name if spec.kind == "llm" else None,
                "backend": spec.backend if spec.kind == "llm" else None,
                "strategy": spec.strategy if spec.kind == "deterministic" else None,
            }
            for spec in lineup
        ],
        "base_config": BASE_CFG,
    }
    write_json(os.path.join(run_dir, "manifest.json"), manifest)

    total = len(m_values) * len(noise_levels) * len(streak_lambdas) * len(seeds)
    completed = 0

    for M in m_values:
        for p in noise_levels:
            for lam in streak_lambdas:
                for seed in seeds:
                    cfg = _make_cfg(
                        base_cfg=BASE_CFG,
                        N=N,
                        M=int(M),
                        p=float(p),
                        lam=float(lam),
                        seed=int(seed),
                    )

                    match_id = _match_id(
                        lineup=lineup,
                        N=N,
                        M=int(M),
                        p=float(p),
                        lam=float(lam),
                        seed=int(seed),
                    )
                    match_dir = ensure_dir(os.path.join(run_dir, match_id))
                    agents_dir = ensure_dir(os.path.join(match_dir, "agents"))

                    match_manifest: Dict[str, Any] = {
                        "run_id": run_id,
                        "match_id": match_id,
                        "seed": int(seed),
                        "N": N,
                        "M": int(M),
                        "noise_p": float(p),
                        "streak_lambda": float(lam),
                        "players": [
                            player_manifest_obj(spec, seat=i)
                            for i, spec in enumerate(lineup)
                        ],
                        "config": cfg,
                    }
                    write_json(
                        os.path.join(match_dir, "match_manifest.json"),
                        match_manifest,
                    )

                    try:
                        agents = [
                            _build_agent(
                                spec=spec,
                                seat=i,
                                cfg=cfg,
                                match_agents_dir=agents_dir,
                            )
                            for i, spec in enumerate(lineup)
                        ]

                        sim = GameSimulator(cfg)
                        result = sim.run_episode(agents)

                        extra_meta: Dict[str, Any] = {
                            "match_id": match_id,
                            "seed": int(seed),
                            "N": N,
                            "M": int(M),
                            "noise_p": float(p),
                            "streak_lambda": float(lam),
                            "lineup_labels": [safe_label(spec) for spec in lineup],
                            "lineup_kinds": [spec.kind for spec in lineup],
                            "lineup_model_ids": [
                                spec.model_name if spec.kind == "llm" else None
                                for spec in lineup
                            ],
                            "lineup_strategies": [
                                spec.strategy if spec.kind == "deterministic" else None
                                for spec in lineup
                            ],
                            "seat_assignment": {
                                str(i): safe_label(spec)
                                for i, spec in enumerate(lineup)
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
                            f"N={N} | M={M} | p={p:.2f} | lam={lam:.2f} | seed={seed} | "
                            f"meta={meta_path} | logs={logs_path}"
                        )

                    except Exception as e:
                        write_json(
                            os.path.join(match_dir, "error.json"),
                            {
                                "run_id": run_id,
                                "match_id": match_id,
                                "seed": int(seed),
                                "N": N,
                                "M": int(M),
                                "noise_p": float(p),
                                "streak_lambda": float(lam),
                                "players": [
                                    player_manifest_obj(spec, seat=i)
                                    for i, spec in enumerate(lineup)
                                ],
                                "config": cfg,
                                "error": repr(e),
                            },
                        )
                        print(f"[ERROR] {match_id}: {e}")

    print("\nDone.")
    print("Run dir:", run_dir)
    return run_dir


def main() -> None:
    run_n_player_engine(
        standard_names=STANDARD,
        llm_names=LLMS,
        m_values=[5],
        noise_levels=[0.0],
        streak_lambdas=[0.0],
        seeds=SEEDS,
    )


if __name__ == "__main__":
    main()