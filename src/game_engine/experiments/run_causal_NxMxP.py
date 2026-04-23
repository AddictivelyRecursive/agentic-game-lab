from __future__ import annotations

import os
from dataclasses import replace
from datetime import datetime
from typing import Any, Dict, List, Sequence

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
from game_engine.io.run_paths import ensure_dir, write_json


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROMPT_DIR = os.path.join(SRC_ROOT, "AI_Agent", "prompts")

# Keep short for Windows
RESULTS_ROOT = os.path.join("results", "nxm")


# ---------------------------------------------------------------------
# Fixed grid
# ---------------------------------------------------------------------

FIXED_T = 50
FIXED_THETA = 0.6
FIXED_STREAK_LAMBDA = 0.25

# Keep same style as current causal runners.
# For smoother 3D surfaces later, change to e.g. [101, 102, 103].
SEEDS: List[int] = [101]

N_VALUES: List[int] = [2, 3, 4, 5]
M_VALUES: List[int] = [2, 3, 5]
NOISE_SLICES: List[float] = [0.00, 0.05, 0.20]

BASE_CFG = EnvConfig(
    N=5,  # overwritten per match
    M=5,  # overwritten per match
    T=FIXED_T,
    p_perception=0.05,  # overwritten per match
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


# ---------------------------------------------------------------------
# Progressive lineup
# ---------------------------------------------------------------------

DEEPSEEK = PlayerSpec(
    kind="llm",
    label="deepseek_v32",
    model_name="deepseek/deepseek-v3.2",
    backend="openrouter",
)

LLAMA = PlayerSpec(
    kind="llm",
    label="llama31_8b",
    model_name="meta-llama/llama-3.1-8b-instruct",
    backend="openrouter",
)

GPT = PlayerSpec(
    kind="llm",
    label="gpt_oss_20b",
    model_name="openai/gpt-oss-20b",
    backend="openrouter",
)

QWEN = PlayerSpec(
    kind="llm",
    label="qwen3_235b_a22b_2507",
    model_name="qwen/qwen3-235b-a22b-2507",
    backend="openrouter",
)

GEMMA = PlayerSpec(
    kind="llm",
    label="gemma3_27b",
    model_name="google/gemma-3-27b-it",
    backend="openrouter",
)

LINEUPS_BY_N: Dict[int, List[PlayerSpec]] = {
    2: [GPT, LLAMA],
    3: [GPT, LLAMA, DEEPSEEK],
    4: [GPT, LLAMA, DEEPSEEK, GEMMA],
    5: [GPT, LLAMA, DEEPSEEK, GEMMA, QWEN],
}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _float_tag(x: float) -> str:
    return f"{float(x):.2f}".replace(".", "p")


def _run_id() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    m_tag = "-".join(str(m) for m in M_VALUES)
    return (
        f"nxm__n2-5"
        f"__m{m_tag}"
        f"__ps{len(NOISE_SLICES)}"
        f"__th{_float_tag(FIXED_THETA)}"
        f"__lam{_float_tag(FIXED_STREAK_LAMBDA)}"
        f"__s{len(SEEDS)}"
        f"__t{FIXED_T}"
        f"__{ts}"
    )


def _match_id(*, noise_p: float, N: int, M: int, seed: int) -> str:
    return f"p{_float_tag(noise_p)}__n{int(N)}__m{int(M)}__s{int(seed)}"


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


def _make_cfg(*, N: int, M: int, noise_p: float, seed: int) -> EnvConfig:
    cfg = replace(
        BASE_CFG,
        N=int(N),
        M=int(M),
        T=FIXED_T,
        p_perception=float(noise_p),
        streak=replace(
            BASE_CFG.streak,
            theta=float(FIXED_THETA),
            lam=float(FIXED_STREAK_LAMBDA),
        ),
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
        raise ValueError(
            f"run_causal_NxM_noise_slices_progressive.py expects LLM-only lineups, got {safe_label(spec)!r}"
        )

    seat_dir = ensure_dir(os.path.join(match_agents_dir, f"p{seat}"))
    return LLMWrapperAgent(
        name=f"p{seat}",
        agent_id=seat,
        env_cfg=cfg,
        backend=spec.backend,
        model_name=spec.model_name or "",
        prompt_dir=PROMPT_DIR,
        output_dir=seat_dir,
        **dict(spec.kwargs or {}),
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    all_specs: List[PlayerSpec] = []
    for N in N_VALUES:
        all_specs.extend(LINEUPS_BY_N[N])

    _require_openrouter_key_if_needed(all_specs)

    run_id = _run_id()
    run_dir = ensure_dir(os.path.join(RESULTS_ROOT, run_id))

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "experiment_type": "causal_NxM_noise_slices_progressive",
        "prompt_dir": PROMPT_DIR,
        "num_matches_expected": len(NOISE_SLICES) * len(N_VALUES) * len(M_VALUES) * len(SEEDS),
        "noise_slices": list(NOISE_SLICES),
        "N_values": list(N_VALUES),
        "M_values": list(M_VALUES),
        "fixed_T": FIXED_T,
        "fixed_theta": FIXED_THETA,
        "fixed_streak_lambda": FIXED_STREAK_LAMBDA,
        "seeds": list(SEEDS),
        "lineups_by_N": {
            str(N): [
                {
                    "label": safe_label(spec),
                    "kind": spec.kind,
                    "model_name": spec.model_name,
                    "backend": spec.backend,
                }
                for spec in LINEUPS_BY_N[N]
            ]
            for N in N_VALUES
        },
        "config_template": BASE_CFG,
    }
    write_json(os.path.join(run_dir, "manifest.json"), manifest)

    total = len(NOISE_SLICES) * len(N_VALUES) * len(M_VALUES) * len(SEEDS)
    completed = 0

    for noise_p in NOISE_SLICES:
        for N in N_VALUES:
            lineup = LINEUPS_BY_N[N]
            lineup_labels = [safe_label(spec) for spec in lineup]

            assert len(lineup_labels) == len(set(lineup_labels)), f"Duplicate labels in N={N} lineup."

            for M in M_VALUES:
                for seed in SEEDS:
                    cfg = _make_cfg(N=N, M=M, noise_p=noise_p, seed=seed)
                    assert cfg.N == len(lineup), f"cfg.N mismatch for N={N}."

                    match_id = _match_id(noise_p=noise_p, N=N, M=M, seed=seed)
                    match_dir = ensure_dir(os.path.join(run_dir, match_id))
                    agents_dir = ensure_dir(os.path.join(match_dir, "agents"))

                    match_manifest: Dict[str, Any] = {
                        "run_id": run_id,
                        "match_id": match_id,
                        "seed": int(seed),
                        "noise_p": float(noise_p),
                        "N": int(N),
                        "M": int(M),
                        "T": int(cfg.T),
                        "theta": float(cfg.streak.theta),
                        "streak_lambda": float(cfg.streak.lam),
                        "lineup_label": f"progressive_N{N}",
                        "player_labels": lineup_labels,
                        "players": [player_manifest_obj(spec, seat=i) for i, spec in enumerate(lineup)],
                        "config": cfg,
                    }
                    write_json(os.path.join(match_dir, "match_manifest.json"), match_manifest)

                    try:
                        agents = [
                            _build_agent(
                                spec=spec,
                                seat=seat,
                                cfg=cfg,
                                match_agents_dir=agents_dir,
                            )
                            for seat, spec in enumerate(lineup)
                        ]

                        sim = GameSimulator(cfg)
                        result = sim.run_episode(agents)

                        extra_meta: Dict[str, Any] = {
                            "match_id": match_id,
                            "seed": int(seed),
                            "noise_p": float(noise_p),
                            "N": int(N),
                            "M": int(M),
                            "T": int(cfg.T),
                            "theta": float(cfg.streak.theta),
                            "streak_lambda": float(cfg.streak.lam),
                            "lineup_label": f"progressive_N{N}",
                            "player_labels": lineup_labels,
                            "player_model_ids": {
                                str(i): lineup[i].model_name for i in range(len(lineup))
                            },
                            "seat_assignment": {
                                str(i): lineup_labels[i] for i in range(len(lineup))
                            },
                            "progressive_growth_rule": (
                                "N=2: GPT+Llama; "
                                "N=3: +DeepSeek; "
                                "N=4: +Gemma; "
                                "N=5: +Qwen"
                            ),
                            "noise_slice": float(noise_p),
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
                            f"p={noise_p:.2f} | N={N} | M={M} | "
                            f"theta={cfg.streak.theta:.2f} | lam={cfg.streak.lam:.2f} | "
                            f"seed={seed} | meta={meta_path} | logs={logs_path}"
                        )

                    except Exception as e:
                        write_json(
                            os.path.join(match_dir, "error.json"),
                            {
                                "run_id": run_id,
                                "match_id": match_id,
                                "seed": int(seed),
                                "noise_p": float(noise_p),
                                "N": int(N),
                                "M": int(M),
                                "T": int(cfg.T),
                                "theta": float(cfg.streak.theta),
                                "streak_lambda": float(cfg.streak.lam),
                                "lineup_label": f"progressive_N{N}",
                                "player_labels": lineup_labels,
                                "error": repr(e),
                            },
                        )
                        print(f"[ERROR] {match_id}: {e}")

    print("\nDone.")
    print("Run dir:", run_dir)


if __name__ == "__main__":
    main()