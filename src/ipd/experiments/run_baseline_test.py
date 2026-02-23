"""
experiments/run_baseline_test.py

Purpose:
- End-to-end sanity test of env with baseline agents.
- Writes JSONL logs for quick inspection and plotting.

Return values:
- Produces output files in ./results/baseline_test/
"""

from __future__ import annotations

import os
from datetime import datetime

from ipd.env.types import EnvConfig, PayoffConfig, DriftConfig, StreakConfig, ObservationConfig
from ipd.env.simulator import GameSimulator

from ipd.agents.baselines import AlwaysCooperate, AlwaysDefect, RandomAgent, GradedTFT
from ipd.io.jsonl import write_episode


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
    # With streak boost max ~ (1+lam), ensure even at B_min:
    # B_min*(1+lam) > C
    assert cfg.payoff.B_min * (1.0 + cfg.streak.lam) > cfg.payoff.C, "PD constraint violated: B_eff may drop <= C"

    agents = [
        GradedTFT("tft0"),
        RandomAgent("rand1"),
        AlwaysDefect("def2"),
        AlwaysCooperate("coop3"),
    ]

    sim = GameSimulator(cfg)
    result = sim.run_episode(agents)

    run_id = "baseline_" + datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_dir = os.path.join("results", "baseline_test")
    meta_path, logs_path = write_episode(out_dir, run_id, 0, result)

    print("Wrote:", meta_path)
    print("Wrote:", logs_path)
    print("Total rewards:", result.total_rewards)
    print("Final B:", result.final_B, "Final streak:", result.final_streak)


if __name__ == "__main__":
    main()