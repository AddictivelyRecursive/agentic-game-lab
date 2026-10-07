"""Package frozen working sources and evidence for the approved fresh main run."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

root=Path(__file__).resolve().parents[1]
out=root/'results/analysis/h4_deadline_budget_20261007/launch_bundle_v3'
out.mkdir()
code=out/'code'
code.mkdir()
for folder in ('src','experiments','cluster','docs'):
    shutil.copytree(root/folder,code/folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc','results','outputs'))
for path in root.glob('*.py'):shutil.copy2(path,code/path.name)
for path in root.glob('requirements*.txt'):shutil.copy2(path,code/path.name)
evidence=code/'evidence'
evidence.mkdir()
shutil.copy2(root/'results/analysis/main_dry_run_review_20261007/run_plan.json',evidence/'old_plan.json')
shutil.copy2(root/'results/analysis/pilot_variance_20261007/variance_estimates.json',evidence/'variance_estimates.json')
provenance=dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
                git_status=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=root,text=True),
                authorization='User 2026-10-07: stop current Qwen; run experiments as planned',
                allocation=dict(zip(['Llama','Qwen','Gemma','Mistral','DeepSeek','GPT-OSS'],[14,4,8,3,16,2])),
                pilot_credit=0)
(evidence/'provenance.json').write_text(json.dumps(provenance,indent=2))
hashes={str(path.relative_to(code)):hashlib.sha256(path.read_bytes()).hexdigest() for path in code.rglob('*') if path.is_file()}
(evidence/'source_hashes.json').write_text(json.dumps(hashes,indent=2))
archive=out.parent/'main_deadline_20261007.tar.gz'
with tarfile.open(archive,'w:gz') as tar:tar.add(code,arcname='code')
print(archive)
