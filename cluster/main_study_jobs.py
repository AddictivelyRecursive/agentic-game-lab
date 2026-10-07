"""Prepare reviewed main plans or explicitly submit independent model shards.
Preparation performs no inference and never invokes sbatch.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.study_common import ROSTER, make_plan, digest, read_json, file_digest


def runner_args(root, plan, count, seed, output):
    return argparse.Namespace(models=list(ROSTER), config=root / "config.json",
        deployment_report=root / "deployment_report.json", generation_overrides=root / "generation_overrides.json",
        power_report=root / "power_report.json", seeds_per_cell=count, seed_start=seed, output=output)


def validate_serving(serving, deployment):
    if type(serving.get("seed_start")) is not int or serving["seed_start"] < 0:
        raise ValueError("Set a nonnegative main seed_start")
    if set(serving.get("models", {})) != set(ROSTER):
        raise ValueError("Serving settings must include all six models")
    from urllib.parse import urlsplit
    for alias, cfg in serving["models"].items():
        if any(v is None for v in cfg.values()):
            raise ValueError("Fill all serving fields for " + alias + "; use empty string for no extra args")
        if type(cfg["tp_size"]) is not int or cfg["tp_size"] < 1:
            raise ValueError("Invalid tensor parallel GPU count")
        if not Path(cfg["model_path"]).is_absolute() or not (Path(cfg["model_path"]) / "config.json").is_file():
            raise ValueError("Verify cached model snapshot: " + alias)
        if urlsplit(deployment["models"][alias]["base_url"]).port != cfg["port"]:
            raise ValueError("Deployment endpoint and serving port differ: " + alias)


def prepare(args):
    bundle = args.bundle.resolve()
    if bundle.exists():
        raise ValueError("Use a new main bundle directory")
    power = read_json(args.power_report)
    count = power.get("seeds_per_cell")
    if type(count) is not int or count < 1:
        raise ValueError("External power calculation must supply seeds_per_cell")
    serving = read_json(args.serving)
    validate_serving(serving, read_json(args.deployment_report))
    bundle.mkdir(parents=True)
    code = bundle / "code"
    code.mkdir()
    # Freeze actual working sources, including uncommitted parser fixes.
    for folder in ("src", "experiments", "cluster"):
        shutil.copytree(ROOT / folder, code / folder,
            ignore=shutil.ignore_patterns("results", "outputs", "__pycache__"))
    for source in ROOT.glob("*.py"):
        shutil.copy2(source, code / source.name)
    (code / 'docs').mkdir(exist_ok=True)
    shutil.copy2(ROOT / 'docs/H4_amendment_20261007.txt', code / 'docs/H4_amendment_20261007.txt')
    for source in ROOT.glob("requirements*.txt"):
        shutil.copy2(source, code / source.name)
    provenance = {"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True),
        "seeds_per_cell": count, "seed_start": serving["seed_start"], "rounds": 30,
        "serving_sha256": file_digest(args.serving)}
    shutil.copy2(args.serving, bundle / "serving.json")
    lib = (code / "cluster/lib.sh").read_text()
    lib = lib.replace("if command -v git >/dev/null; then",
        'if [[ -n "${AGL_SUBMIT_GIT_COMMIT:-}" ]]; then\n        AGL_GIT_COMMIT="$AGL_SUBMIT_GIT_COMMIT"\n        AGL_GIT_STATUS="${AGL_SUBMIT_GIT_STATUS:-}"\n    elif command -v git >/dev/null; then', 1)
    command = [
        '        setsid python "$REPO_ROOT/run_main.py"',
        '            --models ' + ' '.join(ROSTER),
        '            --config "$EXPERIMENT_CONFIG" --deployment-report "$PILOT_DEPLOYMENT_REPORT"',
        '            --generation-overrides "$PILOT_GENERATION_OVERRIDES" --output "$PILOT_OUTPUT"',
        '            --power-report "$MAIN_POWER_REPORT" --seeds-per-cell "$MAIN_SEEDS_PER_CELL"',
        '            --seed-start "$MAIN_SEED_START" --approval "$PILOT_APPROVAL" --execute-model "$MODEL"',
        '            > "$RUN_DIR/experiment.log" 2>&1 &',
    ]
    continuation = " " + chr(92) + chr(10)
    branch = '    if [[ "$kind" == main ]]; then' + chr(10) + continuation.join(command)
    branch += chr(10) + '    elif [[ "$kind" == pilot ]]; then'
    lib = lib.replace('    if [[ "$kind" == pilot ]]; then', branch, 1)
    (bundle / "main_lib.sh").write_bytes(lib.encode())
    wrapper = """#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
set -eo pipefail
source "$HOME/miniconda3/bin/activate" "$CONDA_ENV"
python "$MAIN_BUNDLE/code/cluster/prepare_resume_output.py"
export RUN_DIR="$MAIN_BUNDLE/logs/${MODEL}_${SLURM_JOB_ID}_r${SLURM_RESTART_COUNT:-0}_$(date +%s)"
source "$MAIN_BUNDLE/main_lib.sh"
run_job main
"""
    (bundle / "main.sbatch").write_bytes(wrapper.encode())
    subprocess.run(["bash", "-n", str(bundle / "main.sbatch"), str(bundle / "main_lib.sh")], check=True)
    for alias in ROSTER:
        root = bundle / alias
        root.mkdir()
        for source, name in ((args.config, "config.json"), (args.deployment_report, "deployment_report.json"),
                             (args.generation_overrides, "generation_overrides.json"), (args.power_report, "power_report.json")):
            shutil.copy2(source, root / name)
        # Freeze the exact variance evidence and fix only its relative reference.
        variance = args.power_report.resolve().parent / power["variance_estimates_path"]
        shutil.copy2(variance, root / "variance_estimates.json")
        bound = dict(power, variance_estimates_path="variance_estimates.json")
        (root / "power_report.json").write_text(json.dumps(bound, indent=2))
        cmd = [sys.executable, str(code / "run_main.py"), "--models", *ROSTER,
            "--config", str(root / "config.json"), "--deployment-report", str(root / "deployment_report.json"),
            "--generation-overrides", str(root / "generation_overrides.json"), "--power-report", str(root / "power_report.json"),
            "--seeds-per-cell", str(count), "--seed-start", str(serving["seed_start"]), "--output", str(root / "output"), "--dry-run"]
        subprocess.run(cmd, cwd=code, check=True, stdout=subprocess.DEVNULL)
        plan = read_json(root / "output/run_plan.json")
        (root / "approval.json").write_text(json.dumps({"plan_sha256": digest(plan),
            "confirmed_in_conversation": False, "approval_reference": None}, indent=2))
    (bundle / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print("Prepared six main dry-runs. Review summaries and confirm each approval receipt before submission.")


def submit(args):
    bundle = args.bundle.resolve()
    if (bundle / "submissions.json").exists():
        raise ValueError("Submission ledger exists; refusing duplicate jobs")
    serving = read_json(bundle / "serving.json")
    provenance = read_json(bundle / "provenance.json")
    if file_digest(bundle / "serving.json") != provenance["serving_sha256"]:
        raise ValueError("Serving settings changed since dry-run")
    prepared = []
    for alias in ROSTER:
        root = bundle / alias
        plan = read_json(root / "output/run_plan.json")
        approval = read_json(root / "approval.json")
        if approval.get("confirmed_in_conversation") is not True or not approval.get("approval_reference") or approval.get("plan_sha256") != digest(plan):
            raise ValueError("Missing reviewed approval for " + alias)
        # Validate frozen code and all inputs in the frozen interpreter module.
        validation = "import sys;sys.path.insert(0,sys.argv[1]);from cluster.main_study_jobs import runner_args;from experiments.study_common import make_plan,digest,read_json;from pathlib import Path;r=Path(sys.argv[2]);p=read_json(r/'output/run_plan.json');a=runner_args(r,p,p['seeds_per_cell'],p['seeds'][0],r/'output');assert digest(make_plan(a,'main'))==digest(p),'Plan changed'"
        subprocess.run([sys.executable, "-c", validation, str(bundle / "code"), str(root)], cwd=bundle / "code", check=True)
        cfg = serving["models"][alias]
        env = dict(os.environ, REPO_ROOT=str(bundle / "code"), MAIN_BUNDLE=str(bundle), STUDY_PHASE="main",
            MODEL=alias, MODEL_PATH=cfg["model_path"], TP_SIZE=str(cfg["tp_size"]), PORT=str(cfg["port"]),
            MAX_MODEL_LEN=str(cfg["max_model_len"]), GPU_MEM_UTIL=str(cfg["gpu_mem_util"]),
            HEALTH_WAIT_MIN=str(cfg["health_wait_min"]), CONDA_ENV=cfg["conda_env"], HF_HOME=cfg["hf_home"],
            EXTRA_SERVE_ARGS=cfg["extra_serve_args"], LOCAL_TIMEOUT=str(cfg["local_timeout"]),
            EXPERIMENT_CONFIG=str(root / "config.json"), PILOT_DEPLOYMENT_REPORT=str(root / "deployment_report.json"),
            PILOT_GENERATION_OVERRIDES=str(root / "generation_overrides.json"), PILOT_OUTPUT=str(root / "output"),
            PILOT_APPROVAL=str(root / "approval.json"), MAIN_POWER_REPORT=str(root / "power_report.json"),
            MAIN_SEEDS_PER_CELL=str(provenance["seeds_per_cell"]), MAIN_SEED_START=str(provenance["seed_start"]),
            AGL_SUBMIT_GIT_COMMIT=provenance["git_commit"], AGL_SUBMIT_GIT_STATUS=provenance["git_status"])
        cmd = ["sbatch", "--parsable", "--export=ALL", "--partition="+cfg["partition"], "--qos="+cfg["qos"],
            "--gres=gpu:"+str(cfg["tp_size"]), "--cpus-per-task="+str(cfg["cpus_per_task"]), "--mem="+cfg["memory"],
            "--time="+cfg["walltime"], "--job-name=agl_main_"+alias,
            "--output="+str(bundle / "main_%j.out"), str(bundle / "main.sbatch")]
        prepared.append((alias, cmd, env))
    # No scheduler mutation until every model passes validation.
    submitted = []
    for alias, cmd, env in prepared:
        jid = subprocess.check_output(cmd, env=env, cwd=bundle / "code", text=True).strip().split(";")[0]
        submitted.append({"model": alias, "job_id": jid, "dependency": None})
        (bundle / "submissions.json").write_text(json.dumps(submitted, indent=2))
        print(alias, jid, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "submit"))
    parser.add_argument("--bundle", type=Path, required=True)
    for name in ("config", "deployment-report", "generation-overrides", "power-report", "serving"):
        parser.add_argument("--"+name, type=Path)
    args = parser.parse_args()
    try:
        if args.action == "prepare":
            if any(getattr(args, name.replace("-", "_")) is None for name in ("config", "deployment-report", "generation-overrides", "power-report", "serving")):
                raise ValueError("Preparation requires all five input files")
            prepare(args)
        else:
            submit(args)
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as exc:
        parser.exit(2, str(exc)+"\n")

if __name__ == "__main__":
    main()
