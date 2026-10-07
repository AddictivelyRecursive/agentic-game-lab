"""Execute only the frozen disjoint tasks owned by one Slurm allocation."""
import argparse
import json
import os
from pathlib import Path
import subprocess

p=argparse.ArgumentParser()
p.add_argument('--bundle', type=Path, required=True)
p.add_argument('--run-worker', required=True)
a=p.parse_args()
workers=json.loads((a.bundle/'workers.json').read_text())
worker=next(w for w in workers if w['id']==a.run_worker)
for task in worker['tasks']:
    env=dict(os.environ, **task['environment'])
    env['RUN_DIR']=str(a.bundle/'logs'/f"{a.run_worker}_{task['model']}_{os.environ['SLURM_JOB_ID']}")
    subprocess.run(['bash','-c','source "$MAIN_BUNDLE/main_lib.sh"; run_job main'], env=env, check=True)
