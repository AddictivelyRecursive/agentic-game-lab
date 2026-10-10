"""One-time replacement of our two diagnostics; run only inside the v2 bundle."""
import json
import math
from pathlib import Path
import subprocess
import sys
import time
HERE=Path(__file__).resolve().parent

def command(args):
    return subprocess.check_output(args,text=True).strip()

def main():
    if (HERE/'replacement_receipt.json').exists() or (HERE/'submissions.json').exists(): raise RuntimeError('Already processed; no duplicate jobs')
    for job in ['388834','388835']:
        state=command(['squeue','-h','-j',job,'-o','%i,%j,%T'])
        if state:
            jid,name,status=state.split(',')
            if jid!=job or not name.startswith('agl_cal_'): raise ValueError('Job identity mismatch')
            if status not in ['PENDING','RUNNING']: raise ValueError('Unexpected job state')
            subprocess.run(['scancel',job],check=True)
    elapsed=None
    for _ in range(15):
        records=command(['sacct','-X','-j','388834,388835','--format=JobID,State,ElapsedRaw,AllocTRES','--parsable2','--noheader'])
        parsed=[r.split('|') for r in records.splitlines() if r]
        if len(parsed)==2 and all(r[1].split()[0] in ['CANCELLED','COMPLETED','FAILED','TIMEOUT'] for r in parsed):
            if any('gres/gpu=2' in r[3] for r in parsed): raise ValueError('Unexpected old allocation')
            elapsed=sum(int(r[2]) for r in parsed); break
        time.sleep(2)
    if elapsed is None: raise RuntimeError('Accounting not final; do not resubmit automatically')
    # Reserve two extra minutes for cancellation/accounting uncertainty.
    ds_minutes=math.floor((4*3600-elapsed-120-2*3600)/60)
    if not 30<=ds_minutes<=120: raise RuntimeError('Remaining calibration envelope insufficient')
    receipt=dict(replaced_jobs=['388834','388835'],reason='Original bank used compact JSON, while the original pipeline uses indent=2; Qwen pilot remains archived and non-qualifying. DeepSeek reduced to round 15.',old_allocated_seconds=elapsed,replacement_qwen_minutes=120,replacement_deepseek_minutes=ds_minutes,max_total_gpu_hours=(elapsed+120*60+ds_minutes*60)/3600,accounting=parsed)
    (HERE/'replacement_receipt.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt),flush=True)
    subprocess.run([sys.executable,str(HERE/'submit_calibration.py'),'--submit','--qwen-minutes','120','--deepseek-minutes',str(ds_minutes)],check=True)

if __name__=='__main__': main()
