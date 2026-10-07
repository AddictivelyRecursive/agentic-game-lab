"""Run a disjoint short shard, preserving completed episodes across requeue."""
import argparse
import json
import os
from pathlib import Path
import subprocess

p=argparse.ArgumentParser()
p.add_argument('--bundle',type=Path,required=True)
p.add_argument('--run-worker',required=True)
a=p.parse_args()
workers=json.loads((a.bundle/'workers.json').read_text())
worker=next(w for w in workers if w['id']==a.run_worker)
restart=int(os.environ.get('SLURM_RESTART_COUNT','0'))
for task in worker['tasks']:
    env=dict(os.environ,**task['environment'])
    output=Path(env['PILOT_OUTPUT']).resolve()
    if not output.is_relative_to(a.bundle.resolve()):raise ValueError('Output outside owned bundle')
    raw=output/'main.jsonl'
    rows=[json.loads(line) for line in raw.read_text().splitlines()][1:] if raw.exists() else []
    if any(r['status']!='completed' for r in rows):raise ValueError('Failed episode requires review; no silent retry')
    if len(rows)==len(task['seeds'])*8:continue
    lock=output/'.execution.lock'
    if lock.exists():
        if restart<1:raise ValueError('Unexpected active/stale lock')
        lock.unlink()
    completed={r['episode_id'] for r in rows}
    if restart:
        for folder in (output/'episodes').glob('*'):
            if folder.is_dir() and folder.name not in completed:
                archive=output/'interrupted'/f'restart{restart}'/folder.name
                archive.parent.mkdir(parents=True,exist_ok=True)
                folder.rename(archive)
    env['RUN_DIR']=str(a.bundle/'logs'/f"{a.run_worker}_{os.environ['SLURM_JOB_ID']}_r{restart}")
    subprocess.run(['bash','-c','source "$MAIN_BUNDLE/main_lib.sh"; run_job main'],env=env,check=True)
