"""
experiments/run_baseline_test.py

Purpose:
- End-to-end sanity test of env with baseline agents + optional LLM agents.
- Writes logs using the same directory logic as the cross-play runner.
"""

from __future__ import annotations

import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

from game_engine.agents import AlwaysCooperate, AlwaysDefect, RandomAgent
from game_engine.agents.llm_wrapper import LLMWrapperAgent
from game_engine.env.simulator import GameSimulator
from game_engine.env.types import (
    DriftConfig,
    EnvConfig,
    ObservationConfig,
    PayoffConfig,
    StreakConfig,
)
from game_engine.io.jsonl import write_episode
from game_engine.io.run_paths import build_experiment_id, ensure_dir, write_json


def main() -> None:
    cfg = EnvConfig(
        N=4,
        M=5,
        T=50,
        p_perception=0.10,
        payoff=PayoffConfig(
            B0=2.0,
            C=1.0,
            K=0.0,
            B_min=1.2,
            B_max=3.0,
        ),
        drift=DriftConfig(
            window_w=10,
            eta=0.05,
            r_star=0.5,
        ),
        streak=StreakConfig(
            theta=0.6,
            lam=0.3,
            tau=5.0,
        ),
        obs=ObservationConfig(history_k=10, stats_window=10),
        seed=123,
    )

    assert (
        cfg.payoff.B_min * (1.0 + cfg.streak.lam) > cfg.payoff.C
    ), "PD constraint violated: B_eff may drop <= C"

    run_id = build_experiment_id(
        "baseline",
        cfg=cfg,
        num_seeds=1,
        extra_tags=["smoke"],
    )
    run_dir = ensure_dir(os.path.join("results", "baseline", run_id))
    match_dir = ensure_dir(os.path.join(run_dir, "single_match"))

    write_json(
        os.path.join(run_dir, "manifest.json"),
        {
            "run_id": run_id,
            "experiment_type": "baseline",
            "config": cfg,
        },
    )

    agents = [
        LLMWrapperAgent(
            name="llm0",
            agent_id=0,
            env_cfg=cfg,
            backend="openrouter",
            model_name="openai/gpt-4o-mini",
            prompt_dir="AI_Agent/prompts",
            output_dir=os.path.join(match_dir, "agents", "p0__llm0"),
        ),
        RandomAgent("rand1"),
        AlwaysDefect("def2"),
        AlwaysCooperate("coop3"),
    ]

    write_json(
        os.path.join(match_dir, "match_manifest.json"),
        {
            "run_id": run_id,
            "match_id": "single_match",
            "config": cfg,
            "agents": [
                "openai/gpt-4o-mini",
                "RandomAgent",
                "AlwaysDefect",
                "AlwaysCooperate",
            ],
        },
    )

    sim = GameSimulator(cfg)
    result = sim.run_episode(agents)

    meta_path, logs_path = write_episode(
        match_dir,
        run_id,
        0,
        result,
        extra_meta={"match_id": "single_match"},
        stem="episode_000",
    )

    print("Wrote:", meta_path)
    print("Wrote:", logs_path)
    print("Total rewards:", result.total_rewards)
    print("Final B:", result.final_B, "Final streak:", result.final_streak)
    print("Run dir:", run_dir)


if __name__ == "__main__":
    main()