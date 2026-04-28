from __future__ import annotations

import os
from dataclasses import replace
from typing import Any, Dict, List, Sequence
from datetime import datetime
import hashlib

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

from game_engine.agents import LLMWrapperAgent
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
    player_manifest_obj,
    safe_label,
    validate_social_dilemma_cfg,
)
from game_engine.io.jsonl import write_episode
from game_engine.io.run_paths import build_experiment_id, ensure_dir, write_json

SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROMPT_DIR = os.path.join(SRC_ROOT, "AI_Agent", "prompts")
RESULTS_ROOT = os.path.join(SRC_ROOT, "results", "llama_comparison")

# Llama models to compare
LLAMA_SPECS = [
    PlayerSpec(
        kind="llm",
        label="llama31_8b",
        model_name="meta-llama/llama-3.1-8b-instruct",
        backend="openrouter",
    ),
    # PlayerSpec(
    #     kind="llm",
    #     label="llama32_3b",
    #     model_name="meta-llama/llama-3.2-3b-instruct",
    #     backend="openrouter",
    # ),
    PlayerSpec(
        kind="llm",
        label="llama31_70b",
        model_name="meta-llama/llama-3.1-70b-instruct",
        backend="openrouter",
    ),
]

SEEDS = [101, 102, 103, 104, 105] # 5 seeds
FIXED_T = 50
FIXED_M = 5 # Default number of decisions
FIXED_NOISE_P = 0.05 # Default perception noise
FIXED_THETA = 0.6 # Default streak theta
FIXED_STREAK_LAMBDA = 0.25

BASE_CFG = EnvConfig(
    N=len(LLAMA_SPECS),
    M=FIXED_M,
    T=FIXED_T,
    p_perception=FIXED_NOISE_P,
    payoff=PayoffConfig(
        B0=12.0,
        C=8.0,
        K=0.0,
        B_min=9.0,
        B_max=15.0,
    ),
    drift=DriftConfig(
        window_w=8,
        eta=0.35,
        r_star=0.55,
    ),
    streak=StreakConfig(
        theta=FIXED_THETA,
        lam=FIXED_STREAK_LAMBDA,
        tau=4.0,
    ),
    obs=ObservationConfig(
        history_k=10,
        stats_window=10,
    ),
    seed=0,
)

def _require_openrouter_key_if_needed(specs: Sequence[PlayerSpec]) -> None:
    needs_key = any(
        spec.kind == "llm" and (spec.backend or "").strip().lower() == "openrouter"
        for spec in specs
    )
    if not needs_key:
        return
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise EnvironmentError(
            "OPENROUTER_API_KEY not found. Put it in .env or export it in your shell."
        )

def _make_cfg(*, base_cfg: EnvConfig, seed: int) -> EnvConfig:
    cfg = replace(
        base_cfg,
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
    if spec.kind != "llm":
        raise ValueError(f"Expected LLM player, got {spec.kind}")

    label = safe_label(spec)
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

def main() -> None:
    _require_openrouter_key_if_needed(LLAMA_SPECS)

    run_id = build_experiment_id(
        "llama_comp",
        cfg=BASE_CFG,
        num_seeds=len(SEEDS),
        extra_tags=[
            "llama_family",
            f"m{FIXED_M}",
            f"t{FIXED_T}",
        ],
    )
    run_dir = ensure_dir(os.path.join(RESULTS_ROOT, run_id))

    manifest = {
        "run_id": run_id,
        "experiment_type": "llama_comparison",
        "prompt_dir": PROMPT_DIR,
        "N": len(LLAMA_SPECS),
        "M": FIXED_M,
        "T": FIXED_T,
        "seeds": list(SEEDS),
        "players": [
            {
                "label": safe_label(spec),
                "kind": spec.kind,
                "model_name": spec.model_name,
                "backend": spec.backend,
            }
            for spec in LLAMA_SPECS
        ],
        "config_template": BASE_CFG,
    }
    write_json(os.path.join(run_dir, "manifest.json"), manifest)

    total = len(SEEDS)
    completed = 0

    for seed in SEEDS:
        cfg = _make_cfg(base_cfg=BASE_CFG, seed=seed)

        match_id = f"llama_match__s{seed}"
        match_dir = ensure_dir(os.path.join(run_dir, match_id))
        agents_dir = ensure_dir(os.path.join(match_dir, "agents"))

        match_manifest = {
            "run_id": run_id,
            "match_id": match_id,
            "seed": seed,
            "N": cfg.N,
            "M": cfg.M,
            "T": cfg.T,
            "players": [
                player_manifest_obj(spec, seat=i) for i, spec in enumerate(LLAMA_SPECS)
            ],
            "config": cfg,
        }
        write_json(os.path.join(match_dir, "match_manifest.json"), match_manifest)

        try:
            agents = [
                _build_agent(
                    spec=spec,
                    seat=i,
                    cfg=cfg,
                    match_agents_dir=agents_dir,
                )
                for i, spec in enumerate(LLAMA_SPECS)
            ]

            sim = GameSimulator(cfg)
            result = sim.run_episode(agents)

            extra_meta: Dict[str, Any] = {
                "match_id": match_id,
                "seed": seed,
                "N": cfg.N,
                "M": cfg.M,
                "T": cfg.T,
                "player_labels": [safe_label(spec) for spec in LLAMA_SPECS],
                "player_model_ids": {
                    str(i): spec.model_name for i, spec in enumerate(LLAMA_SPECS)
                },
                "seat_assignment": {
                    str(i): safe_label(spec) for i, spec in enumerate(LLAMA_SPECS)
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
                f"Seed={seed} | "
                f"meta={meta_path} | logs={logs_path}"
            )

        except Exception as e:
            write_json(
                os.path.join(match_dir, "error.json"),
                {
                    "run_id": run_id,
                    "match_id": match_id,
                    "seed": seed,
                    "N": cfg.N,
                    "M": cfg.M,
                    "T": cfg.T,
                    "error": repr(e),
                },
            )
            print(f"[ERROR] {match_id}: {e}")

    print("\nDone.")
    print("Run dir:", run_dir)

if __name__ == "__main__":
    main()
