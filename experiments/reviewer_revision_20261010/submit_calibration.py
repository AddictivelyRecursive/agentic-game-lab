"""Submit only the bounded two-job calibration; never a full experiment grid."""
import argparse
import json
import os
from pathlib import Path
import subprocess

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--submit',action='store_true'); a=ap.parse_args(); bundle=Path(__file__).resolve().parent
    if (bundle/'submissions.json').exists(): raise ValueError('Calibration already submitted; refusing duplicates')
    specs=[('qwen3-8b','Qwen/Qwen3-8B','gpu_h100_4','qos_gpu_h100','96G'),('deepseek-r1-distill-qwen-14b','deepseek-ai/DeepSeek-R1-Distill-Qwen-14B','gpu_h200_8','qos_gpu_h200','128G')]
    (bundle/'runs').mkdir(exist_ok=True); jobs=[]
    for model,repo,partition,qos,memory in specs:
        args=['sbatch','--parsable','--nodes=1','--ntasks=1','--gres=gpu:1','--cpus-per-task=4','--mem='+memory,'--time=02:00:00','--partition='+partition,'--qos='+qos,'--job-name=agl_cal_'+model,'--output='+str(bundle/'runs'/'cal_%j.out'),'--export=ALL',str(bundle/'calibration.sbatch')]
        record=dict(model=model,repo=repo,args=args,requests=24,gpus=1,max_gpu_hours=2)
        if a.submit:
            env=dict(os.environ,BUNDLE=str(bundle),MODEL=model,MODEL_REPO=repo)
            r=subprocess.run(args,env=env,text=True,capture_output=True)
            if r.returncode: print(r.stderr); raise RuntimeError('Submission failed; preserve prior receipts and do not duplicate accepted jobs')
            record['job_id']=r.stdout.strip().split(';')[0]
            if not record['job_id'].isdigit(): raise ValueError('Unexpected scheduler receipt')
        jobs.append(record)
        if a.submit: (bundle/'submissions.json').write_text(json.dumps(jobs,indent=2))
        print(json.dumps(record),flush=True)
    if not a.submit: print('Dry-run only; two jobs, 48 calls, <=4 allocated GPU-hours')

if __name__=='__main__': main()
