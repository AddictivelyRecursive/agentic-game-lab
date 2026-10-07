"""Submission-only helper: keep the authorized short-job queue filled."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import sys

BUNDLE=Path('/home/jagatsesh/f20220604_ts/agentic-game-lab/results/sharanga/main_short_chunks_20261007')
lock=(BUNDLE/'submission_feeder.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
(BUNDLE/'feeder.pid').write_text(str(os.getpid()))
workers=json.loads((BUNDLE/'workers.json').read_text())
while True:
    ledger=json.loads((BUNDLE/'submissions.json').read_text())
    submitted={r['worker'] for r in ledger}
    if len(submitted)==len(workers):
        print('All 29 authorized chunks submitted.',flush=True)
        if '--once' in sys.argv:
            scheduled=subprocess.check_output(['crontab','-l'],text=True)
            remaining='\n'.join(line for line in scheduled.splitlines() if '# AGL_SHORT_MAIN_20261007' not in line)+'\n'
            subprocess.run(['crontab','-'],input=remaining,text=True,check=True)
        break
    try:
        queue=json.loads(subprocess.check_output(['squeue','--json','-u','jagatsesh'],text=True,timeout=30))['jobs']
        usage={q:sum(j.get('qos')==q for j in queue) for q in ('qos_gpu_h100','qos_gpu_h200')}
        pending=[w for w in workers if w['id'] not in submitted]
        # Ensure GPT can enter the H200 queue promptly when a slot opens.
        pending.sort(key=lambda w:(not w['id'].startswith('gpt-oss'),workers.index(w)))
        for worker in pending:
            if usage[worker['qos']]>=6:continue
            env=dict(os.environ,MAIN_BUNDLE=str(BUNDLE),MAIN_WORKER=worker['id'])
            cmd=['sbatch','--parsable','--export=ALL','--nodes=1','--ntasks=1','--gres=gpu:1',
                 '--cpus-per-task=4','--mem='+('128G' if worker['partition']=='gpu_h200_8' else '96G'),
                 '--partition='+worker['partition'],'--qos='+worker['qos'],'--time='+worker['walltime'],
                 '--job-name=agl_short_'+worker['id'],'--output='+str(BUNDLE/'short_%j.out'),str(BUNDLE/'worker.sbatch')]
            result=subprocess.run(cmd,env=env,text=True,capture_output=True,timeout=45)
            if result.returncode:
                print(result.stderr,flush=True)
                continue
            jid=result.stdout.strip().split(';')[0]
            ledger.append(dict(worker=worker['id'],job_id=jid,partition=worker['partition'],dependency=None))
            temporary=BUNDLE/'submissions.tmp'
            temporary.write_text(json.dumps(ledger,indent=2))
            temporary.replace(BUNDLE/'submissions.json')
            usage[worker['qos']]+=1
            print(worker['id'],jid,flush=True)
        (BUNDLE/'feeder_status.json').write_text(json.dumps(dict(submitted=len(ledger),prepared=len(workers),
                                                               checked_at_epoch=time.time(),queue_usage=usage),indent=2))
    except (subprocess.SubprocessError,ValueError,OSError) as exc:
        print(type(exc).__name__,str(exc),flush=True)
    if '--once' in sys.argv:break
    time.sleep(60)
