"""Replace authorized queued long jobs with disjoint short main chunks."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import hashlib

OLD=Path('/home/jagatsesh/f20220604_ts/agentic-game-lab/results/sharanga/main_deadline_20261007_v2')
NEW=OLD.parent/'main_short_chunks_20261007'
PY='/home/jagatsesh/miniconda3/envs/slaybench/bin/python'
REFERENCE='User 2026-10-07: replan, cancel long runs, maximize parallel compute and run short chunks'


def chunks(seeds,size):
    return [seeds[i:i+size] for i in range(0,len(seeds),size)]


def main():
    if NEW.exists():raise ValueError('New bundle exists; refusing duplicate replacement')
    sys.path.insert(0,str(OLD/'code'))
    from experiments.study_common import read_json, digest
    oldworkers=read_json(OLD/'workers.json')
    oldledger=read_json(OLD/'submissions.json')
    oldids=[r['job_id'] for r in oldledger]
    # If any data were collected, stop rather than discard or duplicate them.
    for source in OLD.glob('shards/*/*/output/main.jsonl'):
        if len(source.read_text().splitlines())>1:
            raise ValueError('Existing main episodes require preservation/reconciliation before replacement')
    for jid in oldids:
        data=subprocess.check_output(['scontrol','show','job',jid,'--oneliner'],text=True,timeout=30)
        if 'UserId=jagatsesh(' not in data or 'JobName=agl_main_' not in data or 'JobState=PENDING' not in data:
            raise ValueError('Old job identity/state changed: '+jid)
    NEW.mkdir()
    # Preserve prompts/parser/sampling; add only loopback endpoint routing so
    # concurrent chunks cannot collide on a fixed serving port.
    shutil.copytree(OLD/'code',NEW/'code',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2('/tmp/study_common_short.py',NEW/'code/experiments/study_common.py')
    shutil.copy2('/tmp/consolidate_short.py',NEW/'code/cluster/consolidate_deadline_main.py')
    lib=(OLD/'main_lib.sh').read_text()
    lib=lib.replace('--seeds-per-cell 16 --seed-start 1001', '--runtime-base-url "$LLM_BASE_URL" --seeds-per-cell 16 --seed-start 1001')
    (NEW/'main_lib.sh').write_text(lib)
    (NEW/'worker.sbatch').write_text('''#!/bin/bash
set -eo pipefail
source "$HOME/miniconda3/bin/activate" slaybench
python "$MAIN_BUNDLE/run_worker.py" --bundle "$MAIN_BUNDLE" --run-worker "$MAIN_WORKER"
''')
    shutil.copy2('/tmp/run_short_worker.py',NEW/'run_worker.py')
    grouped={}
    for w in oldworkers:
        for t in w['tasks']:
            entry=grouped.setdefault(t['model'],{'environment':t['environment'],'seeds':[]})
            entry['seeds']+=t['seeds']
    sizes={'llama-3.1-8b-instruct':7,'gemma-3-12b-it':4}
    hours={'llama-3.1-8b-instruct':4,'gemma-3-12b-it':3,
           'qwen3-8b':8,'mistral-small-3.1-24b-instruct':4,
           'deepseek-r1-distill-qwen-14b':12,'gpt-oss-120b':6}
    workers=[]
    for model,entry in grouped.items():
        for index,seeds in enumerate(chunks(sorted(entry['seeds']),sizes.get(model,1))):
            h200=model=='gpt-oss-120b' or index%2==1
            name=f'{model}__chunk{index:02d}'
            root=NEW/'shards'/name/model
            root.mkdir(parents=True)
            oldroot=Path(entry['environment']['PILOT_OUTPUT'])
            env=dict(entry['environment'],REPO_ROOT=str(NEW/'code'),PORT=str(28000+len(workers)),
                     PILOT_OUTPUT=str(root/'output'),PILOT_APPROVAL=str(root/'approval.json'),
                     MAIN_EXECUTE_SEEDS=' '.join(map(str,seeds)))
            cmd=[PY,str(NEW/'code/run_main.py'),'--models',*grouped.keys(),
                 '--config',env['EXPERIMENT_CONFIG'],'--deployment-report',env['PILOT_DEPLOYMENT_REPORT'],
                 '--generation-overrides',env['PILOT_GENERATION_OVERRIDES'],'--power-report',env['MAIN_POWER_REPORT'],
                 '--seeds-per-cell','16','--seed-start','1001','--output',env['PILOT_OUTPUT'],'--dry-run']
            subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL)
            plan=read_json(root/'output/run_plan.json')
            if plan['episode_count']!=376:raise ValueError('Wrong main plan')
            (root/'approval.json').write_text(json.dumps(dict(plan_sha256=digest(plan),confirmed_in_conversation=True,
                                                            approval_reference=REFERENCE),indent=2))
            workers.append(dict(id=name,partition='gpu_h200_8' if h200 else 'gpu_h100_4',
                                qos='qos_gpu_h200' if h200 else 'qos_gpu_h100',
                                walltime=f'{hours[model]:02d}:00:00',
                                tasks=[dict(model=model,seeds=seeds,environment=env)]))
    assigned=[(t['model'],seed) for w in workers for t in w['tasks'] for seed in t['seeds']]
    assert len(assigned)==len(set(assigned))==47
    assert len(workers)==29
    (NEW/'workers.json').write_text(json.dumps(workers,indent=2))
    hashes={str(path.relative_to(NEW/'code')):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (NEW/'code').rglob('*') if path.is_file() and '__pycache__' not in path.parts
            and path.name!='source_hashes.json'}
    (NEW/'code/evidence/source_hashes.json').write_text(json.dumps(hashes,indent=2))
    plans=[read_json(Path(w['tasks'][0]['environment']['PILOT_OUTPUT'])/'run_plan.json') for w in workers]
    assert len({digest(plan) for plan in plans})==1
    subprocess.run(['bash','-n',str(NEW/'worker.sbatch'),str(NEW/'main_lib.sh')],check=True)
    # Cancel ONLY the explicitly identified pending study jobs after all chunks
    # are prepared. Unrelated rb_* jobs are preserved.
    subprocess.run(['scancel',*oldids],check=True,timeout=30)
    (NEW/'cancellation.json').write_text(json.dumps({'cancelled_job_ids':oldids,'approval_reference':REFERENCE},indent=2))
    ledger=[]
    for worker in workers:
        env=dict(os.environ,MAIN_BUNDLE=str(NEW),MAIN_WORKER=worker['id'])
        cmd=['sbatch','--parsable','--export=ALL','--nodes=1','--ntasks=1','--gres=gpu:1',
             '--cpus-per-task=4','--mem='+('128G' if worker['partition']=='gpu_h200_8' else '96G'),
             '--partition='+worker['partition'],'--qos='+worker['qos'],'--time='+worker['walltime'],
             '--job-name=agl_short_'+worker['id'],'--output='+str(NEW/'short_%j.out'),str(NEW/'worker.sbatch')]
        jid=subprocess.check_output(cmd,env=env,text=True,timeout=45).strip().split(';')[0]
        ledger.append(dict(worker=worker['id'],job_id=jid,partition=worker['partition'],dependency=None))
        (NEW/'submissions.json').write_text(json.dumps(ledger,indent=2))
        print(worker['id'],jid,worker['partition'],flush=True)


if __name__=='__main__':main()
