"""Read-only Slurm availability/status snapshots; Slurm allocates queued requests."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import time

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--loop',action='store_true'); a=ap.parse_args()
    here=Path(__file__).resolve().parent
    jobs=json.loads((here/'submissions.json').read_text())
    ids=','.join(j['job_id'] for j in jobs)
    while True:
        record={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
        for name,cmd in [('availability',['sinfo','-N','-o','%N %P %t %G']),('jobs',['squeue','-j',ids,'-o','%.18i %.12P %.10T %.12M %.30R']),('accounting',['sacct','-X','-j',ids,'--format=JobID,State,Elapsed,AllocTRES,ExitCode','--parsable2'])]:
            r=subprocess.run(cmd,capture_output=True,text=True,timeout=30)
            record[name]={'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
        with (here/'monitor.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
        if not a.loop: break
        time.sleep(300)

if __name__=='__main__': main()
