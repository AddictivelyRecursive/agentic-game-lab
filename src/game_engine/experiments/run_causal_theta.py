#checks how hard it is for cooperation streaks to be
# recognized/maintained

from __future__ import annotations

import hashlib
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


SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROMPT_DIR = os.path.join(SRC_ROOT, "AI_Agent", "prompts")
RESULTS_ROOT = os.path.join(SRC_ROOT, "results", "causal_theta")


# Fixed 5-LLM lineup for this study.
LLM_SPECS = [
    PlayerSpec(
        kind="llm",
        label="deepseek_v32",
        model_name="deepseek/deepseek-v3.2",
        backend="openrouter",
    ),
    PlayerSpec(
        kind="llm",
        label="llama31_8b",
        model_name="meta-llama/llama-3.1-8b-instruct",
        backend="openrouter",
    ),
    PlayerSpec(
        kind="llm",
        label="gpt_oss_20b",
        model_name="openai/gpt-oss-20b",
        backend="openrouter",
    ),
    PlayerSpec(
        kind="llm",
        label="qwen3_235b_a22b_2507",
        model_name="qwen/qwen3-235b-a22b-2507",
        backend="openrouter",
    ),
    PlayerSpec(
        kind="llm",
        label="gemma3_27b",
        model_name="google/gemma-3-27b-it",
        backend="openrouter",
    ),
]

THETA_VALUES = [0.10, 0.25, 0.50, 0.75]

FIXED_M = 5
FIXED_NOISE_P = 0.05
FIXED_STREAK_LAMBDA = 0.25
SEEDS = [101]
FIXED_T = 50


BASE_CFG = EnvConfig(
    N=2,
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
        theta=0.25,
        lam=FIXED_STREAK_LAMBDA,
        tau=4.0,
    ),
    obs=ObservationConfig(
        history_k=10,
        stats_window=10,
    ),
    seed=0,
)


LABEL_ALIASES = {
    "deepseek_v32": "dsv32",
    "llama31_8b": "llama31",
    "gpt_oss_20b": "gptoss20",
    "qwen3_235b_a22b_2507": "qwen3235",
    "gemma3_27b": "gemma327",
}


def _sha1_short(text: str, n: int = 6) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


def _short_label(spec: PlayerSpec) -> str:
    raw = safe_label(spec)
    return LABEL_ALIASES.get(raw, raw[:12])


def _lineup_token(lineup: Sequence[PlayerSpec]) -> str:
    short_names = [_short_label(spec) for spec in lineup]
    joined = "-".join(short_names)
    digest = _sha1_short("|".join(safe_label(spec) for spec in lineup), 6)
    return f"{joined}__h{digest}"


def _run_id(
    *,
    N: int,
    M: int,
    theta_values: Sequence[float],
    noise_p: float,
    streak_lambda: float,
    seeds: Sequence[int],
) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    th_tag = "-".join(f"{th:.2f}".replace(".", "p") for th in sorted(set(theta_values)))
    return (
        f"ct__N{N}"
        f"__M{int(M)}"
        f"__th{th_tag}"
        f"__p{float(noise_p):.2f}"
        f"__lam{float(streak_lambda):.2f}"
        f"__seeds{len(seeds)}"
        f"__t{FIXED_T}"
        f"__{ts}"
    )


def _match_id(
    *,
    lineup: Sequence[PlayerSpec],
    N: int,
    M: int,
    theta: float,
    noise_p: float,
    streak_lambda: float,
    seed: int,
) -> str:
    lineup_tok = _lineup_token(lineup)
    th_tag = f"{theta:.2f}".replace(".", "p")
    return (
        f"{lineup_tok}"
        f"__n{N}"
        f"__m{int(M)}"
        f"__th{th_tag}"
        f"__p{float(noise_p):.2f}"
        f"__lam{float(streak_lambda):.2f}"
        f"__s{int(seed)}"
    )


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
    theta: float,
    noise_p: float,
    streak_lambda: float,
    seed: int,
) -> EnvConfig:
    cfg = replace(
        base_cfg,
        N=int(N),
        M=int(M),
        T=FIXED_T,
        p_perception=float(noise_p),
        streak=replace(
            base_cfg.streak,
            theta=float(theta),
            lam=float(streak_lambda),
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
            "run_causal_theta.py expects an all-LLM lineup. "
            f"Found non-LLM player: {safe_label(spec)!r}"
        )

    label = _short_label(spec)
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


def run_causal_theta_engine(
    *,
    llm_specs: Sequence[PlayerSpec],
    theta_values: Sequence[float],
    M: int,
    noise_p: float,
    streak_lambda: float,
    seeds: Sequence[int],
) -> str:
    lineup = list(llm_specs)

    if not lineup:
        raise ValueError("run_causal_theta.py requires a non-empty LLM lineup.")

    for spec in lineup:
        if spec.kind != "llm":
            raise ValueError(
                "run_causal_theta.py expects an all-LLM lineup. "
                f"Found non-LLM player: {safe_label(spec)!r}"
            )

    _require_openrouter_key_if_needed(lineup)

    N = len(lineup)
    run_id = _run_id(
        N=N,
        M=int(M),
        theta_values=theta_values,
        noise_p=noise_p,
        streak_lambda=streak_lambda,
        seeds=seeds,
    )
    run_dir = ensure_dir(os.path.join(RESULTS_ROOT, run_id))

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "experiment_type": "causal_theta",
        "prompt_dir": PROMPT_DIR,
        "fixed_T": FIXED_T,
        "selected_llms": [safe_label(spec) for spec in lineup],
        "N": N,
        "fixed_M": int(M),
        "theta_values": list(theta_values),
        "noise_p": float(noise_p),
        "streak_lambda": float(streak_lambda),
        "seeds": list(seeds),
        "lineup_labels": [safe_label(spec) for spec in lineup],
        "lineup_short_labels": [_short_label(spec) for spec in lineup],
        "lineup_token": _lineup_token(lineup),
        "lineup_players": [
            {
                "label": safe_label(spec),
                "short_label": _short_label(spec),
                "kind": spec.kind,
                "model_name": spec.model_name,
                "backend": spec.backend,
                "strategy": None,
            }
            for spec in lineup
        ],
        "base_config": BASE_CFG,
    }
    write_json(os.path.join(run_dir, "manifest.json"), manifest)

    total = len(theta_values) * len(seeds)
    completed = 0

    for theta in theta_values:
        for seed in seeds:
            cfg = _make_cfg(
                base_cfg=BASE_CFG,
                N=N,
                M=int(M),
                theta=float(theta),
                noise_p=float(noise_p),
                streak_lambda=float(streak_lambda),
                seed=int(seed),
            )

            match_id = _match_id(
                lineup=lineup,
                N=N,
                M=int(M),
                theta=float(theta),
                noise_p=float(noise_p),
                streak_lambda=float(streak_lambda),
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
                "theta": float(theta),
                "noise_p": float(noise_p),
                "streak_lambda": float(streak_lambda),
                "lineup_token": _lineup_token(lineup),
                "players": [
                    player_manifest_obj(spec, seat=i) for i, spec in enumerate(lineup)
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
                    for i, spec in enumerate(lineup)
                ]

                sim = GameSimulator(cfg)
                result = sim.run_episode(agents)

                extra_meta: Dict[str, Any] = {
                    "match_id": match_id,
                    "seed": int(seed),
                    "N": N,
                    "M": int(M),
                    "theta": float(theta),
                    "noise_p": float(noise_p),
                    "streak_lambda": float(streak_lambda),
                    "lineup_labels": [safe_label(spec) for spec in lineup],
                    "lineup_short_labels": [_short_label(spec) for spec in lineup],
                    "lineup_kinds": [spec.kind for spec in lineup],
                    "lineup_model_ids": [spec.model_name for spec in lineup],
                    "lineup_strategies": [None for _ in lineup],
                    "seat_assignment": {
                        str(i): _short_label(spec) for i, spec in enumerate(lineup)
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
                    f"N={N} | M={M} | theta={theta:.2f} | p={noise_p:.2f} | "
                    f"lam={streak_lambda:.2f} | seed={seed} | "
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
                        "theta": float(theta),
                        "noise_p": float(noise_p),
                        "streak_lambda": float(streak_lambda),
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
    run_causal_theta_engine(
        llm_specs=LLM_SPECS,
        theta_values=THETA_VALUES,
        M=FIXED_M,
        noise_p=FIXED_NOISE_P,
        streak_lambda=FIXED_STREAK_LAMBDA,
        seeds=SEEDS,
    )


if __name__ == "__main__":
    main()