"""POST-HOC / EXPLORATORY: six short, dependent CPU jobs; zero GPUs/inference.

Slurm dispatches jobs when resources become available. No polling daemon, cron,
GPU queries, model server, or inference entrypoint is installed by this helper.
"""
import argparse
import datetime
import json
import shlex
import subprocess
import sys
from pathlib import Path

def main():
    here=Path(__file__).resolve().parent
    ap=argparse.ArgumentParser(); ap.add_argument('--input',type=Path,required=True); ap.add_argument('--export-dir',type=Path,required=True); ap.add_argument('--python',type=Path,default=here/'.venv_posthoc_20261010/bin/python'); ap.add_argument('--partition',default='compute'); ap.add_argument('--qos',default='cpulimit'); ap.add_argument('--cpus',type=int,default=4); a=ap.parse_args()
    for path in [a.input,a.export_dir/'coding_sheet.csv',a.export_dir/'private_condition_key.json',a.python]:
        if not path.exists(): raise FileNotFoundError(path)
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'); log=here/'logs'/stamp; log.mkdir(parents=True,exist_ok=False)
    jobs=[]; previous=None
    for script in ['00_reproduction.py','01_robustness.py','02_heterogeneity.py','03_persistence.py','04_mismatch.py','finalize.py']:
        command=[str(a.python),str(here/script)]
        if script!='finalize.py': command+=['--input',str(a.input)]
        if script=='04_mismatch.py': command+=['--export-dir',str(a.export_dir)]
        args=['sbatch','--parsable','--partition='+a.partition,'--cpus-per-task='+str(a.cpus),'--mem=4G','--time=00:10:00','--job-name=agl_posthoc_'+script[:2],'--output='+str(log/(script+'.out')),'--chdir='+str(here.parent),'--export=ALL,OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1']
        if a.qos: args+=['--qos='+a.qos]
        if previous: args+=['--dependency=afterok:'+previous,'--kill-on-invalid-dep=yes']
        args+=['--wrap='+shlex.join(command)]
        result=subprocess.run(args,capture_output=True,text=True)
        if result.returncode:
            print(result.stderr,file=sys.stderr); raise SystemExit(result.returncode)
        receipt=result.stdout.strip(); previous=receipt.split(';')[0]
        if not previous.isdigit(): raise RuntimeError('Unexpected scheduler receipt: '+receipt)
        jobs.append(dict(script=script,job_id=previous,command=command,resources=dict(cpus=a.cpus,memory='4G',minutes=10,gpus=0)))
        (log/'submissions.json').write_text(json.dumps(jobs,indent=2)); print(script,previous,flush=True)
    print('Receipt:',log/'submissions.json')

if __name__=='__main__': main()
