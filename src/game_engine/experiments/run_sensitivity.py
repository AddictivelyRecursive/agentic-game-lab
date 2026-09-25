"""Controlled paired decision scenarios. Run from src with python -m game_engine.experiments.run_sensitivity."""
from game_engine.experiments.settings import ROUNDS
import argparse
import json
from copy import deepcopy
from pathlib import Path

from AI_Agent.agent.llm_agent import LLMAgent
from AI_Agent.agent.openrouter_client import OpenRouterClient


def make_turn(N=2, M=5, p=0.0, eta=0.0):
    # Preserve endpoint cooperation histories across action granularities.
    rows = [[0, 0, M-1, 0], [M-1, 0, 0, 0]]
    history = [rows[0]] + [list(rows[1]) for _ in range(N-1)]
    return {
        "round": 5, "agent_id": 0,
        "temporal_context": {"total_rounds": ROUNDS, "rounds_remaining_including_current": ROUNDS - 4},
        "game_parameters": {"N": N, "M": M, "perception_noise_p": p,
            "action_semantics": {"index_to_cooperation": [1-i/(M-1) for i in range(M)]}},
        "payoff": {"B_base": 12., "B_effective": 12., "C": 8., "K": 0.,
                   "formula": "u_i = B_eff * avg_other_coop(true) - C * own_coop(true) + K"},
        "streak_rule": {"theta": .6, "tau": 4., "lambda": 0., "streak_prev": 0,
                        "streak_update_rule": "Increment if observed global mean cooperation >= theta; otherwise reset to zero"},
        "streak_effect_on_payoff": {"equation": "B_eff = B_base * (1 + lambda * tanh(streak_prev/tau))"},
        "drift_rule": {"window_w": 4, "eta": eta, "r_star": .55,
                       "bounds": {"B_min": 9., "B_max": 15.},
                       "update_equation": "B_next = clip(B_base + eta*(rolling_observed_global_cooperation-r_star), B_min, B_max)"},
        "information_set": {"observed_history_last_k": history,
                            "rolling_observed_cooperation_prev": .75},
    }


def scenarios():
    return [(dimension, level, make_turn(**changes)) for dimension, variants in {
        "N": [(2, {}), (5, {"N": 5})],
        "M": [(2, {"M": 2}), (5, {})],
        "p": [(0., {}), (.4, {"p": .4})],
        "X_eta": [(0., {}), (.35, {"eta": .35})],
    }.items() for level, changes in variants]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1 or (not args.dry_run and not args.model):
        parser.error("Use positive repeats and provide --model unless --dry-run")
    args.output.mkdir(parents=True, exist_ok=False)
    cases = scenarios()
    manifest = {"design": "paired_snapshot_v1", "models": args.model, "repeats": args.repeats,
                "memory_mode": "history_only", "predict_opponents": False,
                "max_retries": 3, "scenarios": cases,
                "notes": "Hypothetical supplied observations, not sampled trajectories. N duplicates opponent history; M preserves cooperation levels. X varies eta only, with streak bonus disabled. Repeats are unseeded model samples."}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if args.dry_run:
        return
    with (args.output / "decisions.jsonl").open("w") as output:
        for index, model in enumerate(args.model):
            agent = LLMAgent(llm_client=OpenRouterClient(model),
                prompt_dir=str(Path(__file__).parents[2] / "AI_Agent/prompts"),
                output_dir=str(args.output / str(index)))
            for repeat in range(args.repeats):
                for dimension, level, turn in cases:
                    agent.reset()
                    action = agent.step(deepcopy(turn))
                    state = agent.last_state
                    record = dict(model=model, repeat=repeat, dimension=dimension, level=level,
                        action=action, cooperation=turn["game_parameters"]["action_semantics"]["index_to_cooperation"][action],
                        valid_model_decision=state["valid_model_decision"], decision=state["decision"])
                    output.write(json.dumps(record) + "\n")
                    output.flush()
                    if not state["valid_model_decision"]:
                        raise RuntimeError("Model failure: saved diagnostics; stopping to avoid contaminated comparisons")


if __name__ == "__main__":
    main()
