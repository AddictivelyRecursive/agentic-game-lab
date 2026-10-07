"""One fresh full-model rerun for fallback-contaminated pilot shards.
Retain original results; no selective episode replacement or retry-until-pass.
"""
import json
from pathlib import Path
import subprocess
import sys
repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo))
from experiments.study_common import digest
old = repo / "results/sharanga/pilot30_temp07_20261003"
root = repo / "results/sharanga/pilot30_quality_rerun_20261004"
root.mkdir(exist_ok=False)
for name in ("study_config.json", "deployment_report.json", "generation_overrides.json"):
    value = json.loads((old / name).read_text())
    (root / name).write_text(json.dumps(value, indent=2) + "\n")
models = json.loads((old / "output/run_plan.json").read_text())["models"]
subprocess.run([sys.executable, str(repo / "run_pilot.py"), "--models", *models,
    "--config", str(root / "study_config.json"), "--deployment-report", str(root / "deployment_report.json"),
    "--generation-overrides", str(root / "generation_overrides.json"), "--output", str(root / "output"),
    "--seed-start", "106", "--dry-run"], cwd=repo, check=True)
plan = json.loads((root / "output/run_plan.json").read_text())
assert all(c["generation"]["temperature"] == 0.7 and c["generation"]["top_p"] == 1.0 for c in plan["cells"])
(root / "approval.json").write_text(json.dumps({"plan_sha256": digest(plan), "confirmed_in_conversation": True,
    "approval_reference": "2026-10-04 user: put the failed ones to run again - whatever is unacceptable for publishing; full contaminated-model rerun with fresh seeds 106-110, originals retained"}, indent=2) + "\n")
script = (old / "submit_pilot.py").read_text().replace("pilot30_temp07_20261003", "pilot30_quality_rerun_20261004")
start = script.index("sequence=")
end = script.index("\nlog=", start)
script = script[:start] + "sequence=[('llama-3.1-8b-instruct','03:00:00'),('qwen3-8b','30:00:00')]" + script[end:]
script = script.replace("submitted=[];previous=None", "submitted=[];previous='378646'")
script = script.replace("--dependency=afterok:", "--dependency=afterany:")
# A separate job script passes fresh seeds, preserving the active runner and its plan.
script = script.replace("cluster/sharanga_pilot.sbatch", "cluster/sharanga_quality_rerun.sbatch")
(root / "submit_pilot.py").write_text(script)
print("Prepared", root, "plan", digest(plan))
