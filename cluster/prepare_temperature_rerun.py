"""Prepare a fresh, approved temperature-only pilot rerun on Sharanga; no inference."""
import json
from pathlib import Path
import subprocess
import sys

repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo))
from experiments.study_common import digest
old = repo / "results/sharanga/pilot30_20261003"
root = repo / "results/sharanga/pilot30_temp07_20261003"
root.mkdir(exist_ok=False)
for name in ("study_config.json", "deployment_report.json", "generation_overrides.json"):
    value = json.loads((old / name).read_text())
    if name == "study_config.json":
        value["generation"].update(temperature=0.7, top_p=1.0)
    (root / name).write_text(json.dumps(value, indent=2) + "\n")
models = json.loads((old / "output/run_plan.json").read_text())["models"]
command = [sys.executable, str(repo / "run_pilot.py"), "--models", *models,
    "--config", str(root / "study_config.json"), "--deployment-report", str(root / "deployment_report.json"),
    "--generation-overrides", str(root / "generation_overrides.json"), "--output", str(root / "output"), "--dry-run"]
subprocess.run(command, cwd=repo, check=True)
plan = json.loads((root / "output/run_plan.json").read_text())
assert all(c["generation"]["temperature"] == 0.7 and c["generation"]["top_p"] == 1.0 for c in plan["cells"])
(root / "approval.json").write_text(json.dumps({"plan_sha256": digest(plan), "confirmed_in_conversation": True,
    "approval_reference": "2026-10-03 user explicitly requested temperature 0.7/top_p 1.0 before submission and rerunning all completed and failed Sharanga pilots"}, indent=2) + "\n")
script = (old / "submit_pilot.py").read_text().replace("pilot30_20261003", "pilot30_temp07_20261003")
# Startup timeout was the previous Mistral failure; allow loading to finish.
script = script.replace("env['AGL_SUBMIT_GIT_COMMIT']", "env['HEALTH_WAIT_MIN']='120'\n env['AGL_SUBMIT_GIT_COMMIT']")
(root / "submit_pilot.py").write_text(script)
print("Prepared", root, "plan", digest(plan))
