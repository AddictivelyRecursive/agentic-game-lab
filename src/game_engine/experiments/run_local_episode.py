"""Run configured episodes through the existing simulator against local vLLM.

This is a single-configuration infrastructure entrypoint, not the Phase 2 sweep.
"""
import argparse
import json
import math
import os
from pathlib import Path

from AI_Agent.agent.local_openai_client import LocalOpenAIClient
from game_engine.agents.llm_wrapper import LLMWrapperAgent
from game_engine.env.simulator import GameSimulator
from game_engine.env.types import EnvConfig, PayoffConfig, DriftConfig, StreakConfig, ObservationConfig
from game_engine.io.jsonl import write_episode
from game_engine.io.run_paths import write_json


def load_config(path):
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(spec) != {"environment", "seeds", "agent", "generation"}:
        raise ValueError("Config requires exactly environment, seeds, agent, generation")
    environment = dict(spec["environment"])
    if "seed" in environment:
        raise ValueError("Specify episode seeds in seeds, not environment.seed")
    for key, cls in (("payoff", PayoffConfig), ("drift", DriftConfig),
                     ("streak", StreakConfig), ("obs", ObservationConfig)):
        environment[key] = cls(**environment[key])
    seeds = spec["seeds"]
    if not isinstance(seeds, list) or not seeds or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("seeds must be a nonempty list of nonnegative integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate episode seeds are not allowed")
    cfg = EnvConfig(**environment, seed=seeds[0])
    if any(type(x) is not int for x in (cfg.N, cfg.M, cfg.T)) or cfg.N < 2 or not 2 <= cfg.M <= 10 or cfg.T < 1:
        raise ValueError("Require N >= 2, 2 <= M <= 10, T >= 1, all integers")
    # The legacy causal-study validator enforces T=50. This infrastructure runner
    # explicitly permits short smoke episodes; it does not alter that validator.
    numeric = (cfg.p_perception, cfg.payoff.B0, cfg.payoff.C, cfg.payoff.K,
               cfg.payoff.B_min, cfg.payoff.B_max, cfg.streak.theta,
               cfg.streak.lam, cfg.streak.tau, cfg.drift.eta, cfg.drift.r_star)
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in numeric):
        raise ValueError("Environment parameters must be finite numbers")
    if not (0 <= cfg.p_perception <= 1 and 0 <= cfg.streak.theta <= 1):
        raise ValueError("Require perception probability and theta in [0, 1]")
    if not (0 < cfg.payoff.C < cfg.payoff.B_min <= cfg.payoff.B0 <= cfg.payoff.B_max):
        raise ValueError("Require 0 < C < B_min <= B0 <= B_max")
    if cfg.streak.lam < 0 or cfg.streak.tau <= 0 or cfg.drift.eta < 0 or not 0 <= cfg.drift.r_star <= 1:
        raise ValueError("Invalid streak or drift configuration")
    windows = [cfg.obs.history_k, cfg.drift.window_w]
    if cfg.obs.stats_window is not None:
        windows.append(cfg.obs.stats_window)
    if any(type(x) is not int or x < 1 for x in windows):
        raise ValueError("History, statistics and drift windows must be positive integers")
    agent = spec["agent"]
    if set(agent) != {"memory_mode", "predict_opponents", "reasoning_mode", "max_retries"}:
        raise ValueError("Specify memory_mode, predict_opponents, reasoning_mode, max_retries")
    if agent["reasoning_mode"] not in ("structured", "free_form") or agent["memory_mode"] not in ("history_only", "model_memory"):
        raise ValueError("Invalid reasoning or memory mode")
    if type(agent["predict_opponents"]) is not bool or type(agent["max_retries"]) is not int or agent["max_retries"] < 0:
        raise ValueError("Invalid prediction flag or repair budget")
    if set(spec["generation"]) != {"temperature", "top_p", "max_tokens", "seed"}:
        raise ValueError("Explicit generation temperature, top_p, max_tokens, seed (or null) required")
    # Validate transport-independent generation settings without issuing any request.
    LocalOpenAIClient("validation", "http://127.0.0.1:1/v1", generation_config=spec["generation"])
    return spec, environment


def run(spec, environment, *, model, base_url, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    run_id = output.name
    manifest = {
        "experiment_type": "local_same_model_episodes", "run_id": run_id,
        "model": model, "base_url": base_url, "configuration": spec,
        "job_manifest": os.getenv("JOB_MANIFEST"),
        "local_timeout_s": os.getenv("LOCAL_TIMEOUT", "600"),
        "lineup": "All N seats use the selected model, with separate agent state",
    }
    write_json(str(output / "manifest.json"), manifest)
    try:
        for seed in spec["seeds"]:
            cfg = EnvConfig(**environment, seed=seed)
            episode_dir = output / f"seed_{seed}"
            agents = [LLMWrapperAgent(
                name=f"{model}_p{seat}", agent_id=seat, env_cfg=cfg,
                backend="local", model_name=model, base_url=base_url,
                generation_config=spec["generation"], stop_on_api_failure=True,
                prompt_dir=str(Path(__file__).resolve().parents[2] / "AI_Agent" / "prompts"),
                output_dir=str(episode_dir / "agents" / f"p{seat}"), **spec["agent"],
            ) for seat in range(cfg.N)]
            result = GameSimulator(cfg).run_episode(agents)
            paths = write_episode(str(episode_dir), run_id, seed, result,
                                  extra_meta={"model": model, "generation": spec["generation"]})
            print("Saved:", *paths, flush=True)
            if any(meta.fallback_used for row in result.logs for meta in row.agent_meta):
                raise RuntimeError("Episode contains fallback decisions; results retained, run failed")
    except Exception as exc:
        write_json(str(output / "error.json"), {"error": str(exc), "type": type(exc).__name__})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", default=os.getenv("LLM_MODEL"))
    parser.add_argument("--base-url", default=os.getenv("LLM_BASE_URL"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    spec, environment = load_config(args.config)
    if args.validate_only:
        print(json.dumps(spec, indent=2))
        return
    if not args.model or not args.base_url or args.output is None:
        parser.error("--model/LLM_MODEL, --base-url/LLM_BASE_URL and --output are required")
    run(spec, environment, model=args.model, base_url=args.base_url, output=args.output)


if __name__ == "__main__":
    main()
