"""
experiments/run_baseline_test.py

Purpose:
- End-to-end sanity test of env with baseline agents + optional LLM agents.
- Writes JSONL logs for quick inspection and plotting.

Return values:
- Produces output files under ./results/baseline_test/<run_id>/
  - episode meta/logs from write_episode(...)
  - per-agent LLM traces (if LLMWrapperAgent used) under:
      results/baseline_test/<run_id>/agents/<agent_name>/
"""

from __future__ import annotations

import os
from datetime import datetime

from game_engine.env.types import EnvConfig, PayoffConfig, DriftConfig, StreakConfig, ObservationConfig
from game_engine.env.simulator import GameSimulator

from game_engine.agents import AlwaysCooperate, AlwaysDefect, RandomAgent, GradedTFT
from game_engine.agents.llm_wrapper import LLMWrapperAgent
from game_engine.io.jsonl import write_episode


def main() -> None:
    # Small N for baseline validation; scale later.
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

    # Critical sanity: ensure PD-ish incentives don't break catastrophically.
    assert cfg.payoff.B_min * (1.0 + cfg.streak.lam) > cfg.payoff.C, "PD constraint violated: B_eff may drop <= C"

    # Per-run id + output root
    run_id = "baseline_" + datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_dir = os.path.join("results", run_id)
    os.makedirs(out_dir, exist_ok=True)
    
    import json
    # Save config for reproducibility
    config_path = os.path.join(out_dir, "config.json")

    with open(config_path, "w") as f:
        json.dump(cfg.__dict__, f, indent=2, default=lambda o: o.__dict__)
    
    # --- Agents ---
    # NOTE: N must match cfg.N.
    # Can switch backends per agent:
    #   backend="dummy"  (fast, free; good for sanity)
    #   backend="ollama" (local ollama server; model_name selects model tag)
    agents = [
        LLMWrapperAgent(
            name="llm0",
            agent_id=0,
            env_cfg=cfg,
            backend="dummy",              # change to "ollama" to use real local model
            model_name="llama3.1:8b",     # used when backend="ollama"
            prompt_dir="AI_Agent/prompts",
            output_dir=os.path.join(out_dir, "agents", "llm0"),
            # Dummy controls (ignored by ollama backend)
            dummy_mode="mostly_valid",
            dummy_seed=7,
            dummy_invalid_rate=0.6,
            dummy_force_invalid_first_n6=1,
        ),
        RandomAgent("rand1"),
        AlwaysDefect("def2"),
        AlwaysCooperate("coop3"),
        # Example: model-vs-model (uncomment and also adjust cfg.N/agents list accordingly)
        # LLMWrapperAgent(
        #     name="llm1",
        #     agent_id=1,
        #     env_cfg=cfg,
        #     backend="ollama",
        #     model_name="mistral:7b",
        #     prompt_dir="AI_Agent/prompts",
        #     output_dir=os.path.join(out_dir, "agents", "llm1"),
        # ),
    ]

    sim = GameSimulator(cfg)
    result = sim.run_episode(agents)

    # Episode JSONL logs (env-side)
    meta_path, logs_path = write_episode(out_dir, run_id, 0, result)

    print("Wrote:", meta_path)
    print("Wrote:", logs_path)
    print("Total rewards:", result.total_rewards)
    print("Final B:", result.final_B, "Final streak:", result.final_streak)
    print("Run dir:", out_dir)
    print("Per-agent logs (if any):", os.path.join(out_dir, "agents"))


if __name__ == "__main__":
    main()