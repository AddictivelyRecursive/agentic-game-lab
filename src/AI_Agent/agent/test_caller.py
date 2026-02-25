import json
import os
from typing import Dict, Any

# Import your agent
from AI_Agent.agent.llm_agent import LLMAgent

# Use dummy client
from AI_Agent.agent.dummy_llm_client import DummyLLMClient


def build_sample_turn() -> Dict[str, Any]:
    """
    Builds a minimal valid turn input matching your schema.
    (Fixes trailing commas and ensures JSON-parsable structure.)
    """
    return {
        "round": 12,
        "agent_id": 0,
        "game_parameters": {
            "N": 4,
            "M": 5,
            "perception_noise_p": 0.1,
            "action_semantics": {
                "index_to_cooperation": [1.0, 0.75, 0.5, 0.25, 0.0],
                "note": "Lower index = more cooperation"
            }
        },
        "payoff": {
            "B_base": 2.1,
            "B_effective": 2.35,
            "C": 1.0,
            "K": 0.0,
            "formula": "u_i = B_eff * avg_other_coop(true) - C * own_coop(true) + K"
        },
        "streak_rule": {
            "computed_from": "observed_global_mean_coop",
            "theta": 0.6,
            "tau": 5.0,
            "lambda": 0.3,
            "streak_prev": 3,
            "streak_update_rule": "if mean_obs_coop >= theta then streak_next=streak_prev+1 else 0"
        },
        "streak_effect_on_payoff": {
            "affects": "B_effective",
            "equation": "B_eff = B_base * (1 + lambda * tanh(streak_prev/tau))",
            "interpretation": "Higher streak_prev increases B_eff"
        },
        "drift_rule": {
            "type": "q3_global",
            "window_w": 20,
            "eta": 0.02,
            "r_star": 0.5,
            "r_obs_prev": 0.68,
            "update_equation": "B_next = clip(B_base + eta*(r_obs_current - r_star), [B_min, B_max])",
            "bounds": {"B_min": 1.2, "B_max": 3.0},
            "note": "r_obs_current computed after this round"
        },
        "information_set": {
            # first index is most recent, per your clarification
            "observed_history_last_k": [
                [4, 4, 3, 3, 2, 4, 4, 4, 3, 3],
                [3, 4, 4, 3, 2, 4, 3, 4, 3, 4],
                [4, 4, 4, 3, 3, 3, 4, 4, 4, 4],
                [2, 3, 4, 3, 3, 4, 3, 4, 4, 4]
            ],
            "rolling_observed_cooperation_prev": 0.68,
            "action_freq_NxM": [
                [0.02, 0.08, 0.10, 0.25, 0.55],
                [0.10, 0.20, 0.30, 0.20, 0.20],
                [0.00, 0.05, 0.10, 0.15, 0.70],
                [0.40, 0.20, 0.10, 0.10, 0.20]
            ]
        },
        "instruction": {
            "task": "Choose an integer action a in [0,4] to maximize your long-term average reward.",
            "output_format": "{\"a\": <int>}"
        }
    }


def main():
    # Create agent
    agent = LLMAgent(model_name="dummy", prompt_dir="AI_Agent/prompts")

    # Plug in dummy LLM client instead of Ollama client
    # Mode choices:
    # - always_valid: should pass without retries
    # - mostly_valid: triggers repair sometimes
    # - always_invalid: forces fallback after retries
    agent.llm_client = DummyLLMClient(mode="mostly_valid", seed=7, invalid_rate=0.6, force_invalid_first_n6=1)

    # Run single turn
    turn = build_sample_turn()
    action = agent.step(turn)
    print("Chosen action:", action)

    # Show output files written
    out_dir = "AI_Agent/outputs"
    print("Wrote:", os.path.join(out_dir, "llm_outputs.jsonl"))
    print("Wrote:", os.path.join(out_dir, "agent_traces.jsonl"))

    # Print last trace line for quick inspection
    trace_path = os.path.join(out_dir, "agent_traces.jsonl")
    with open(trace_path, "r") as f:
        last = None
        for line in f:
            last = line
        if last:
            obj = json.loads(last)
            print("Last trace summary:")
            print("  round:", obj.get("round"))
            print("  final_action:", obj.get("final_action"))
            print("  retries:", obj.get("retries"))
            print("  validation_error:", obj.get("validation_error", None))
            print("  nodes:", [t["node"] for t in obj.get("trace", [])])


if __name__ == "__main__":
    main()