from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence

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
from game_engine.io.jsonl import write_episode
from game_engine.io.run_paths import (
    build_experiment_id,
    ensure_dir,
    slugify,
    write_json,
)


# ---------------------------------------------------------------------
# Specs
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class PlayerSpec:
    kind: str  # "llm"
    label: str
    model_name: Optional[str] = None
    backend: str = "openrouter"
    kwargs: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROMPT_DIR = "AI_Agent/prompts"
RESULTS_ROOT = os.path.join("results", "causal_N_progressive")


# ---------------------------------------------------------------------
# Fixed config
# ---------------------------------------------------------------------

BASE_CFG = EnvConfig(
    N=5,  # replaced per match
    M=5,
    T=50,
    p_perception=0.05,
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
        theta=0.6,
        lam=0.25,
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
)

LLAMA = PlayerSpec(
    kind="llm",
    label="llama31_8b",
    model_name="meta-llama/llama-3.1-8b-instruct",
)

GPT = PlayerSpec(
    kind="llm",
    label="gpt_oss_20b",
    model_name="openai/gpt-oss-20b",
)

QWEN = PlayerSpec(
    kind="llm",
    label="qwen3_235b_a22b_2507",
    model_name="qwen/qwen3-235b-a22b-2507",
)

GEMMA = PlayerSpec(
    kind="llm",
    label="gemma3_27b",
    model_name="google/gemma-3-27b-it",
)

LINEUPS_BY_N: Dict[int, List[PlayerSpec]] = {
    2: [DEEPSEEK, LLAMA],
    3: [DEEPSEEK, LLAMA, GPT],
    4: [DEEPSEEK, LLAMA, GPT, QWEN],
    5: [DEEPSEEK, LLAMA, GPT, QWEN, GEMMA],
}

N_VALUES: List[int] = [2, 3, 4, 5]

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


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


def _player_manifest_obj(spec: PlayerSpec, seat: int) -> Dict[str, Any]:
    return {
        "label": _spec_label(spec),
        "kind": spec.kind,
        "model_name": spec.model_name,
        "backend": spec.backend,
        "kwargs": spec.kwargs,
        "seat": seat,
    }


def _make_cfg(*, N: int) -> EnvConfig:
    return replace(BASE_CFG, N=int(N), M=5, T=50, seed=0)


def _build_agent(
    *,
    spec: PlayerSpec,
    seat: int,
    cfg: EnvConfig,
    match_agents_dir: str,
):
    label = _spec_label(spec)

    if spec.kind != "llm":
        raise ValueError(f"Unknown player kind: {spec.kind!r}")

    seat_dir = ensure_dir(os.path.join(match_agents_dir, f"p{seat}__{label}"))
    return LLMWrapperAgent(
        name=f"p{seat}__{label}",
        agent_id=seat,
        env_cfg=cfg,
        backend=spec.backend,
        model_name=spec.model_name or "",
        prompt_dir=PROMPT_DIR,
        output_dir=seat_dir,
        **spec.kwargs,
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    all_specs: List[PlayerSpec] = []
    for N in N_VALUES:
        all_specs.extend(LINEUPS_BY_N[N])
    _require_openrouter_key_if_needed(all_specs)

    run_id = build_experiment_id(
        "cnp",
        cfg=BASE_CFG,
        num_seeds=1,
        extra_tags=[
            "n2-3-4-5",
            "progressive",
            f"m{5}",
            f"t{5}",
        ],
    )
    run_dir = ensure_dir(os.path.join(RESULTS_ROOT, run_id))

    manifest = {
        "run_id": run_id,
        "experiment_type": "causal_N_progressive",
        "prompt_dir": PROMPT_DIR,
        "num_matches_expected": len(N_VALUES),
        "N_values": N_VALUES,
        "M": 5,
        "T": 50,
        "lineups_by_N": {
            str(N): [
                {
                    "label": _spec_label(spec),
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

    completed = 0
    total = len(N_VALUES)

    for N in N_VALUES:
        lineup = LINEUPS_BY_N[N]
        cfg = _make_cfg(N=N)

        lineup_labels = [_spec_label(spec) for spec in lineup]
        assert len(lineup_labels) == len(set(lineup_labels)), f"Duplicate labels in N={N} lineup."
        assert cfg.N == len(lineup), f"cfg.N mismatch for N={N}."

        match_id = slugify(
            f"progressive__n{N}__m{cfg.M}__t{cfg.T}",
            max_len=96,
        )
        match_dir = ensure_dir(os.path.join(run_dir, match_id))
        agents_dir = ensure_dir(os.path.join(match_dir, "agents"))

        match_manifest = {
            "run_id": run_id,
            "match_id": match_id,
            "N": cfg.N,
            "M": cfg.M,
            "T": cfg.T,
            "lineup_label": f"progressive_N{N}",
            "player_labels": lineup_labels,
            "players": [
                _player_manifest_obj(spec, seat=i) for i, spec in enumerate(lineup)
            ],
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
                "N": cfg.N,
                "M": cfg.M,
                "T": cfg.T,
                "lineup_label": f"progressive_N{N}",
                "player_labels": lineup_labels,
                "player_model_ids": {
                    str(i): lineup[i].model_name for i in range(len(lineup))
                },
                "seat_assignment": {
                    str(i): lineup_labels[i] for i in range(len(lineup))
                },
                "progressive_growth_rule": (
                    "N=2: DeepSeek+Llama; "
                    "N=3: +GPT; "
                    "N=4: +Qwen; "
                    "N=5: +Gemma"
                ),
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
                f"N={N} | M={cfg.M} | T={cfg.T} | "
                f"meta={meta_path} | logs={logs_path}"
            )

        except Exception as e:
            write_json(
                os.path.join(match_dir, "error.json"),
                {
                    "run_id": run_id,
                    "match_id": match_id,
                    "N": cfg.N,
                    "M": cfg.M,
                    "T": cfg.T,
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