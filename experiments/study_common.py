"""Shared study contracts. Importing this module never calls a model or server."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from AI_Agent.agent.local_openai_client import LocalOpenAIClient
from AI_Agent.model_registry import resolve_model
from game_engine.experiments.run_local_episode import load_config

ROSTER = (
    "llama-3.1-8b-instruct", "qwen3-8b", "gemma-3-12b-it",
    "mistral-small-3.1-24b-instruct", "deepseek-r1-distill-qwen-14b", "gpt-oss-120b",
)
REASONING_MODELS = {"deepseek-r1-distill-qwen-14b", "gpt-oss-120b"}
MODES = ("structured", "free_form")
N_LEVELS = (2, 3, 4, 5)
TRAINING_NOTE = ("DeepSeek-R1-Distill-Qwen-14B is a DeepSeek-trained distillation on a Qwen base, "
                 "representing an independent training process/organization. Model behavior is empirical.")
SCHEMA = "scaling_scaffolding_v1"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def positive(value, label):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return value


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def validate_models(models):
    if len(models) != 6 or set(models) != set(ROSTER):
        raise ValueError("Pass all six distinct study aliases via --models; roster changes need a revised specification")


def new_output(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def code_fingerprint():
    paths = list((ROOT / "src/AI_Agent").rglob("*.py"))
    paths += list((ROOT / "src/game_engine/env").glob("*.py"))
    paths += list((ROOT / "src/AI_Agent/prompts").glob("*.txt"))
    paths += [ROOT / "src/game_engine/agents/llm_wrapper.py", Path(__file__),
              ROOT / "run_pilot.py", ROOT / "run_main.py",
              ROOT / "src/game_engine/experiments/run_local_episode.py"]
    return digest({str(p.relative_to(ROOT)): file_digest(p) for p in sorted(set(paths))})


def validate_deployment(report, models):
    if report.get("step0_complete") is not True or not report.get("checked_at"):
        raise ValueError("Step 0 is incomplete; supply measured deployment evidence before a study dry-run")
    records = report.get("models", {})
    for alias in models:
        record = records.get(alias, {})
        if record.get("reachable") is not True or not str(record.get("response", "")).strip():
            raise ValueError(f"Missing successful trivial LocalOpenAIClient response: {alias}")
        if record.get("hf_identifier") != resolve_model(alias)["repo"]:
            raise ValueError(f"Deployment model identity mismatch: {alias}")
        if not record.get("evidence_path"):
            raise ValueError(f"Missing deployment evidence reference: {alias}")
        LocalOpenAIClient(alias, record.get("base_url"), timeout_s=1)
        positive(record.get("mean_decision_latency_s"), f"{alias} decision latency")
        positive(record.get("gpu_count"), f"{alias} GPU count")
    gpt = records["gpt-oss-120b"]
    positive(gpt.get("vram_used_gib"), "GPT-OSS measured VRAM")
    if not gpt.get("quantization") or gpt.get("allocation") not in ("separate", "coexists") or not gpt.get("allocation_notes"):
        raise ValueError("GPT-OSS needs quantization and allocation/coexistence evidence")
    smoke = gpt.get("isolated_smoke", {})
    if any(smoke.get(k) != v for k, v in {"N": 2, "T": 5, "completed": True, "fallback_decisions": 0}.items()):
        raise ValueError("GPT-OSS needs a completed isolated N=2, T=5 smoke without fallback")
    if len(smoke.get("round_latency_s", [])) != 5:
        raise ValueError("GPT-OSS needs five measured round latencies")
    for latency in smoke["round_latency_s"]:
        positive(latency, "GPT-OSS round latency")


def make_plan(args, phase):
    validate_models(args.models)
    deployment = read_json(args.deployment_report)
    validate_deployment(deployment, args.models)
    spec, _ = load_config(args.config)
    env = spec["environment"]
    if (env["M"], env["p_perception"], env["streak"]["theta"], env["T"]) != (5, 0, .5, 50):
        raise ValueError("Study requires M=5, p=0, theta=0.50, T=50")
    if spec["agent"]["memory_mode"] != "history_only" or spec["agent"]["predict_opponents"]:
        raise ValueError("This study freezes history_only memory and no opponent prediction")
    count = 5 if phase == "pilot" else args.seeds_per_cell
    power = None
    if phase == "main":
        power = read_json(args.power_report)
        variance_path = Path(args.power_report).parent / power.get("variance_estimates_path", "")
        variance = read_json(variance_path)
        if (power.get("seeds_per_cell") != count or power.get("confirmed_in_conversation") is not True
                or not power.get("approval_reference") or not power.get("rationale")
                or power.get("variance_sha256") != file_digest(variance_path)
                or variance.get("ready_for_power_calculation") is not True):
            raise ValueError("Main requires an explicitly confirmed external power calculation bound to ready pilot variance estimates")
        if variance.get("configuration") != spec or variance.get("models") != args.models:
            raise ValueError("Main configuration/model order differs from the pilot used for power; review the design before proceeding")
    seeds = list(range(args.seed_start, args.seed_start + count))
    if args.seed_start < 0:
        raise ValueError("seed-start must be nonnegative")
    cells = []
    for model, mode, n in itertools.product(args.models, MODES, N_LEVELS):
        measurement = deployment["models"][model]
        seconds = measurement["mean_decision_latency_s"]
        if model == "gpt-oss-120b":
            seconds = statistics.mean(measurement["isolated_smoke"]["round_latency_s"]) / 2
        decisions = n * 50 * count
        cells.append({"model": model, "reasoning_mode": mode, "N": n, "seeds": seeds,
                      "episodes": count, "decisions": decisions, "estimated_seconds": seconds * decisions,
                      "estimated_gpu_hours": seconds * decisions * measurement["gpu_count"] / 3600})
    return {"schema": SCHEMA, "phase": phase, "models": args.models, "seeds_per_cell": count,
            "seeds": seeds, "configuration": spec, "cells": cells, "episode_count": 48 * count,
            "decision_count": sum(c["decisions"] for c in cells), "deployment": deployment,
            "power_calculation": power, "code_sha256": code_fingerprint(),
            "lineup": "All N seats use the cell's model, with separate agent state; no heterogeneous opponents",
            "analysis_unit": "Episode mean normalized true cooperation across all seats and 50 rounds",
            "analysis_choices": {"primary_family": "Gaussian", "primary_link": "identity", "estimation": "ML",
                                 "primary_formula": "cooperation ~ N * free_mode", "reference_mode": "structured",
                                 "random_effects": "model intercept plus N slope if usable convergence, otherwise intercept",
                                 "inference": "95% Wald CI, two-sided asymptotic p-values, alpha=0.05 per test, unadjusted",
                                 "H2": "pooled structured episode Gaussian ML, intercept vs intercept + b/(N-1)",
                                 "H5": "coder-specific rates and N differences after kappa>=0.7; no unspecified inferential test"},
            "generation_seed_policy": "SHA256 of phase/model/mode/N/episode-seed/seat; first 32 bits, used for every request in that seat",
            "fallback_policy": "Retain all trajectories and flag; downstream inference requires zero fallback or a separately reviewed revision",
            "training_note": TRAINING_NOTE}


def plan_text(plan):
    seconds = sum(c["estimated_seconds"] for c in plan["cells"])
    gpu_hours = sum(c["estimated_gpu_hours"] for c in plan["cells"])
    lines = [f"# {plan['phase'].title()} dry-run", f"Plan SHA256: `{digest(plan)}`",
             f"Episodes: {plan['episode_count']}; cells: 48; decisions before repairs: {plan['decision_count']}.",
             f"Estimated serial decision time: {seconds / 3600:.3f} hours; GPU time: {gpu_hours:.3f} GPU-hours.",
             "Estimate scales measured Step 0 latency by seat decisions, using GPT-OSS's five-round smoke latency / 2.",
             "It excludes queue/model loading time; longer histories, mode differences, and repairs can change latency.",
             "GPU-hours count active decision time, not idle reserved endpoints. Monetary cost is not estimated.",
             plan["lineup"], "No inference has run. Explicit conversational approval of this exact plan is required.",
             "", "| Model | Mode | N | Seeds | Episodes | Decision hours |", "|---|---|---:|---|---:|---:|"]
    for c in plan["cells"]:
        lines.append(f"| {c['model']} | {c['reasoning_mode']} | {c['N']} | {','.join(map(str, c['seeds']))} | {c['episodes']} | {c['estimated_seconds']/3600:.3f} |")
    lines += ["", "## Frozen configuration", "```json", json.dumps(plan["configuration"], indent=2), "```",
              "", "## Analysis choices for review", "```json", json.dumps(plan["analysis_choices"], indent=2), "```", "", TRAINING_NOTE]
    return "\n".join(lines) + "\n"


def cell_key(row):
    return row["model"], row["reasoning_mode"], row["N"]


def read_study(path, phase):
    with Path(path).open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    if not records or records[0].get("record_type") != "manifest":
        raise ValueError("Expected study JSONL with a manifest as the first record")
    manifest, rows = records[0], records[1:]
    plan = manifest["plan"]
    if plan.get("schema") != SCHEMA or plan.get("phase") != phase or manifest.get("plan_sha256") != digest(plan):
        raise ValueError("Wrong study phase/schema or corrupted plan")
    validate_models(plan["models"])
    wanted_cells = set(itertools.product(plan["models"], MODES, N_LEVELS))
    if len(plan["cells"]) != 48 or {cell_key(c) for c in plan["cells"]} != wanted_cells:
        raise ValueError("Manifest must specify the full 48-cell grid")
    if type(plan["seeds_per_cell"]) is not int or plan["seeds_per_cell"] < 1:
        raise ValueError("Invalid seed count in manifest")
    if phase == "pilot" and plan["seeds_per_cell"] != 5:
        raise ValueError("Pilot requires exactly five seeds per cell")
    for c in plan["cells"]:
        if (c["episodes"] != plan["seeds_per_cell"] or len(c["seeds"]) != c["episodes"]
                or len(set(c["seeds"])) != c["episodes"] or c["seeds"] != plan["seeds"]):
            raise ValueError("Invalid per-cell seed plan")
    env = plan["configuration"]["environment"]
    if (env["M"], env["T"], env["p_perception"], env["streak"]["theta"]) != (5, 50, 0, .5):
        raise ValueError("Wrong fixed study environment")
    expected = {(c["model"], c["reasoning_mode"], c["N"], s) for c in plan["cells"] for s in c["seeds"]}
    seen = set()
    for row in rows:
        key = (*cell_key(row), row["seed"])
        if row.get("record_type") != "episode" or key not in expected or key in seen:
            raise ValueError(f"Unexpected or duplicate episode: {key}")
        seen.add(key)
        if row["status"] not in ("completed", "failed"):
            raise ValueError("Invalid episode status")
        decisions = row["decisions"]
        keys = [(d["round"], d["seat"]) for d in decisions]
        if len(set(keys)) != len(keys):
            raise ValueError("Duplicate round/seat decisions")
        for d in decisions:
            if not isinstance(d["reasoning_text"], str):
                raise ValueError("Reasoning text must be a string")
            if any(type(d[k]) is not bool for k in ("first_parse_ok", "repair_used", "fallback_used")):
                raise ValueError("Reliability flags must be booleans")
        if row["status"] == "completed":
            wanted = set(itertools.product(range(1, 51), range(row["N"])))
            if set(keys) != wanted or len(row["rounds"]) != 50:
                raise ValueError("Completed episode is missing rounds/seats")
            if [r["t"] for r in row["rounds"]] != list(range(1, 51)):
                raise ValueError("Invalid round ordering")
            for r in row["rounds"]:
                if len(r["true_coop"]) != row["N"] or any(
                        type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in r["true_coop"]):
                    raise ValueError("Invalid raw cooperation values")
            mean = statistics.mean(v for r in row["rounds"] for v in r["true_coop"])
            if not 0 <= row["cooperation"] <= 1 or not math.isclose(mean, row["cooperation"], abs_tol=1e-12):
                raise ValueError("Episode cooperation is inconsistent with raw rounds")
    return manifest, rows


def summarize_cells(plan, rows):
    groups = defaultdict(list)
    for row in rows:
        groups[cell_key(row)].append(row)
    cells = []
    for cell in plan["cells"]:
        attempts = groups[cell_key(cell)]
        completed = [r for r in attempts if r["status"] == "completed"]
        values = [r["cooperation"] for r in completed]
        decisions = [d for r in attempts for d in r["decisions"]]
        fallback = sum(d["fallback_used"] for d in decisions)
        uniform = len(values) >= 2 and max(values) - min(values) <= 1e-12
        rate = fallback / len(decisions) if decisions else None
        flags = []
        if len(completed) != cell["episodes"]:
            flags.append("INCOMPLETE")
        if rate is not None and rate > .1:
            flags.append("FALLBACK_RATE_GT_0.1")
        if fallback:
            flags.append("FALLBACK_CONTAMINATION")
        if uniform:
            flags.append("UNIFORM_ACROSS_SEEDS")
        cells.append({"model": cell["model"], "reasoning_mode": cell["reasoning_mode"], "N": cell["N"],
                      "target": cell["episodes"], "completed": len(completed), "failed": len(attempts) - len(completed),
                      "mean_cooperation": statistics.mean(values) if values else None,
                      "variance": statistics.variance(values) if len(values) > 1 else None,
                      "decision_count": len(decisions), "fallback_rate": rate, "uniform": uniform, "flags": flags})
    return cells


def cell_table(cells):
    lines = ["| Model | Mode | N | Completed/target | Variance (ddof=1) | Fallback rate | Flags |",
             "|---|---|---:|---:|---:|---:|---|"]
    for c in cells:
        fmt = lambda v: "unavailable" if v is None else f"{v:.6g}"
        lines.append(f"| {c['model']} | {c['reasoning_mode']} | {c['N']} | {c['completed']}/{c['target']} | {fmt(c['variance'])} | {fmt(c['fallback_rate'])} | {', '.join(c['flags']) or 'none'} |")
    return "\n".join(lines)


def ready(cells):
    return all(not c["flags"] and c["variance"] is not None for c in cells)


def reliability(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["model"], row["reasoning_mode"]].extend(row["decisions"])
    lines = ["| Model | Mode | Decisions | First-call JSON parse rate | Repair rate | Fallback rate |",
             "|---|---|---:|---:|---:|---:|"]
    for model, mode in itertools.product(ROSTER, MODES):
        ds = groups[model, mode]
        rates = [sum(bool(d[k]) for d in ds) / len(ds) if ds else None
                 for k in ("first_parse_ok", "repair_used", "fallback_used")]
        lines.append(f"| {model} | {mode} | {len(ds)} | " + " | ".join("unavailable" if v is None else f"{v:.5f}" for v in rates) + " |")
    return "\n".join(lines)


def write_run_summary(output, plan, rows):
    cells = summarize_cells(plan, rows)
    phase = plan["phase"]
    lines = [f"# {phase.title()} run summary", cell_table(cells), "", "## Reliability", reliability(rows),
             "", "First-call parse means a JSON object was parsed; it does not imply a valid action.",
             "Repair rate is the fraction of decisions using one or more repair calls.",
             "Fallbacks are retained and flagged, never silently treated as model behavior."]
    if phase == "pilot":
        lines += [f"ready for power calculation: {'yes' if ready(cells) else 'no'}",
                  "Variance estimation only. No hypothesis tests were performed."]
    else:
        complete = all(c["completed"] == c["target"] for c in cells)
        valid = complete and all(c["fallback_rate"] == 0 for c in cells)
        lines += [f"All target seed counts complete: {'yes' if complete else 'no'}",
                  f"Ready for analysis and coding: {'yes' if valid else 'no'}"]
    lines += [TRAINING_NOTE]
    filename = "pilot_summary.md" if phase == "pilot" else "main_run_summary.md"
    (Path(output) / filename).write_text("\n\n".join(lines) + "\n", encoding="utf-8")
    return cells


def first_parse_ok(calls):
    try:
        raw = calls[0]["output"]
        return isinstance(json.loads(raw[raw.find("{"):raw.rfind("}") + 1]), dict)
    except (IndexError, KeyError, ValueError, TypeError):
        return False


def execute_episode(plan, cell, seed, output):
    from game_engine.agents.llm_wrapper import LLMWrapperAgent
    from game_engine.env.simulator import GameSimulator
    from game_engine.env.types import EnvConfig, PayoffConfig, DriftConfig, StreakConfig, ObservationConfig
    from game_engine.io.jsonl import write_episode

    spec = plan["configuration"]
    env = dict(spec["environment"], N=cell["N"], seed=seed)
    for k, cls in (("payoff", PayoffConfig), ("drift", DriftConfig), ("streak", StreakConfig), ("obs", ObservationConfig)):
        env[k] = cls(**env[k])
    cfg = EnvConfig(**env)
    episode_id = f"{cell['model']}__{cell['reasoning_mode']}__n{cfg.N}__seed{seed}"
    episode_dir = Path(output) / "episodes" / episode_id
    episode_dir.mkdir(parents=True, exist_ok=False)
    decisions = []

    class RecordingAgent(LLMWrapperAgent):
        def act(self, obs):
            start = time.perf_counter()
            meta = None
            try:
                action, meta = super().act(obs)
                return action, meta
            finally:
                state = self.llm.last_state
                fallback = bool(state.get("fallback_used", False))
                reason = (state.get("decision") or {}).get("reason", "") if not fallback else ""
                decisions.append({"round": obs.t, "seat": self.agent_id,
                                  "reasoning_text": reason if isinstance(reason, str) else "",
                                  "raw_response": state.get("last_model_output", ""),
                                  "model_calls": state.get("model_calls", []),
                                  "first_parse_ok": first_parse_ok(state.get("model_calls", [])),
                                  "repair_used": state.get("retries", 0) > 0,
                                  "repair_calls": state.get("retries", 0), "fallback_used": fallback,
                                  "aborted": meta is None, "latency_s": time.perf_counter() - start})

    row = {"record_type": "episode", "episode_id": episode_id, "model": cell["model"],
           "reasoning_mode": cell["reasoning_mode"], "N": cfg.N, "seed": seed,
           "status": "failed", "cooperation": None, "decisions": decisions, "rounds": [], "generation_by_seat": []}
    start = time.perf_counter()
    try:
        agents = []
        for seat in range(cfg.N):
            generation = dict(spec["generation"], seed=int(digest([plan["phase"], episode_id, seat])[:8], 16))
            row["generation_by_seat"].append(generation)
            agents.append(RecordingAgent(
                name=f"p{seat}", agent_id=seat, env_cfg=cfg, backend="local", model_name=cell["model"],
                base_url=plan["deployment"]["models"][cell["model"]]["base_url"], generation_config=generation,
                output_dir=str(episode_dir / "agents" / f"p{seat}"), prompt_dir=str(ROOT / "src/AI_Agent/prompts"),
                stop_on_api_failure=True, **dict(spec["agent"], reasoning_mode=cell["reasoning_mode"])))
        result = GameSimulator(cfg).run_episode(agents)
        write_episode(str(episode_dir), plan["phase"], seed, result, extra_meta={
            "model": cell["model"], "reasoning_mode": cell["reasoning_mode"], "generation_by_seat": row["generation_by_seat"]})
        row.update(status="completed", rounds=[asdict(r) for r in result.logs],
                   cooperation=statistics.mean(v for r in result.logs for v in r.true_coop))
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    row["elapsed_s"] = time.perf_counter() - start
    write_json(episode_dir / "study_episode.json", row)
    return row


def run_cli(phase, argv=None):
    parser = argparse.ArgumentParser(description=f"{phase.title()} study grid; no server provisioning or job submission.")
    parser.add_argument("--models", nargs=6, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "experiments/study_config.json")
    parser.add_argument("--deployment-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=101)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--approval", type=Path, help="Human approval receipt bound to the saved dry-run plan")
    parser.add_argument("--execute-model", choices=ROSTER, help="Optional serial shard; full grid is still planned and reported")
    if phase == "main":
        parser.add_argument("--seeds-per-cell", "--seeds_per_cell", type=positive_int, required=True)
        parser.add_argument("--power-report", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        plan = make_plan(args, phase)
        if args.dry_run:
            output = new_output(args.output)
            write_json(output / "run_plan.json", plan)
            text = plan_text(plan)
            (output / "dry_run_summary.md").write_text(text, encoding="utf-8")
            write_run_summary(output, plan, [])
            print(text)
            return 0
        saved = read_json(args.output / "run_plan.json")
        if digest(plan) != digest(saved):
            raise ValueError("Plan differs from saved dry-run (including code/evidence). Generate and review a new dry-run")
        if args.approval is None:
            raise ValueError("Execution requires explicit conversational approval recorded in --approval")
        approval = read_json(args.approval)
        if (approval.get("plan_sha256") != digest(plan) or approval.get("confirmed_in_conversation") is not True
                or not approval.get("approval_reference")):
            raise ValueError("Approval is missing, unconfirmed, or belongs to a different plan")
        lock = args.output / ".execution.lock"
        with lock.open("x", encoding="utf-8") as stream:
            stream.write("Exclusive runner lock. Do not run shards concurrently.\n")
        rows = []
        try:
            raw = args.output / f"{phase}.jsonl"
            if raw.exists():
                manifest, rows = read_study(raw, phase)
                if manifest["plan_sha256"] != digest(plan):
                    raise ValueError("Existing data belongs to another plan")
            else:
                with raw.open("x", encoding="utf-8") as stream:
                    stream.write(json.dumps({"record_type": "manifest", "plan": plan, "plan_sha256": digest(plan), "approval": approval}) + "\n")
            completed = {(*cell_key(r), r["seed"]) for r in rows}
            for cell in plan["cells"]:
                if args.execute_model and cell["model"] != args.execute_model:
                    continue
                for seed in cell["seeds"]:
                    if (*cell_key(cell), seed) in completed:
                        continue
                    row = execute_episode(plan, cell, seed, args.output)
                    with raw.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    rows.append(row)
                    write_run_summary(args.output, plan, rows)
                    print(f"{row['episode_id']}: {row['status']}", flush=True)
                    if row["status"] == "failed":
                        raise RuntimeError(row["error"] + "; partial outputs retained; no automatic retries")
        finally:
            write_run_summary(args.output, plan, rows)
            lock.unlink()
        print(f"Summary written to {args.output}")
        return 0 if all(r["status"] == "completed" for r in rows) else 1
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        parser.exit(2, f"{exc}\n")


def require_complete(manifest, rows):
    cells = summarize_cells(manifest["plan"], rows)
    if any(c["completed"] != c["target"] or c["fallback_rate"] != 0 for c in cells):
        raise ValueError("Main data must contain all 48 cells at their target seed counts and zero fallback; inspect main_run_summary.md")
    return cells
