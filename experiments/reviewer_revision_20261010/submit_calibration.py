"""Submit only the bounded two-job calibration; never a full experiment grid."""
import argparse
import json
import os
from pathlib import Path
import subprocess

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--submit',action='store_true'); ap.add_argument('--models',nargs='+',choices=['qwen3-8b','deepseek-r1-distill-qwen-14b']); ap.add_argument('--qwen-minutes',type=int,default=120); ap.add_argument('--deepseek-minutes',type=int,default=120); a=ap.parse_args(); bundle=Path(__file__).resolve().parent
    if not (1<=a.qwen_minutes<=120 and 1<=a.deepseek_minutes<=120): raise ValueError('Job limits must be 1..120 minutes')
    if (bundle/'submissions.json').exists(): raise ValueError('Calibration already submitted; refusing duplicates')
    specs=[('qwen3-8b','Qwen/Qwen3-8B','gpu_h100_4','qos_gpu_h100','96G'),('deepseek-r1-distill-qwen-14b','deepseek-ai/DeepSeek-R1-Distill-Qwen-14B','gpu_h200_8','qos_gpu_h200','128G')]
    (bundle/'runs').mkdir(exist_ok=True); jobs=[]
    for model,repo,partition,qos,memory in specs:
        if a.models and model not in a.models: continue
        minutes=a.qwen_minutes if model=='qwen3-8b' else a.deepseek_minutes
        args=['sbatch','--parsable','--nodes=1','--ntasks=1','--gres=gpu:1','--cpus-per-task=4','--mem='+memory,'--time='+str(minutes),'--partition='+partition,'--qos='+qos,'--job-name=agl_cal_'+model,'--output='+str(bundle/'runs'/'cal_%j.out'),'--export=ALL',str(bundle/'calibration.sbatch')]
        record=dict(model=model,repo=repo,args=args,requests=24 if model=='qwen3-8b' else 8,gpus=1,max_gpu_hours=minutes/60)
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
