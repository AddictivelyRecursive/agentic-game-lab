import json, os, subprocess
from pathlib import Path
base=Path('/home/jagatsesh/f20220604_ts/agentic-game-lab')
bundle=base/'results/sharanga/parallel_parser_v2_20261005'
repo=bundle/'code'
old=base/'results/sharanga/pilot30_temp07_20261003'
prior=json.loads((old/'output/run_plan.json').read_text())
os.environ['AGL_SUBMIT_GIT_COMMIT']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
os.environ['AGL_SUBMIT_GIT_STATUS']=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=repo,text=True)
prepared=[]
for alias,seed,duration in [('gpt-oss-120b',101,'20:00:00'),('llama-3.1-8b-instruct',106,'03:00:00'),('qwen3-8b',106,'30:00:00')]:
    root=bundle/alias
    assert not (root/'output/pilot.jsonl').exists()
    record=prior['deployment']['models'][alias];job=json.loads((Path(record['evidence_path'])/'job.json').read_text());cfg=job['environment']
    env=dict(os.environ,REPO_ROOT=str(repo),MODEL=alias,MODEL_REPO=cfg['MODEL_REPO'],MODEL_PATH=job['vllm_command'][2],SERVED_NAME=alias,TP_SIZE='1',PORT=cfg['PORT'],MAX_MODEL_LEN=cfg['MAX_MODEL_LEN'],GPU_MEM_UTIL=cfg['GPU_MEM_UTIL'],CONDA_ENV=cfg['CONDA_ENV'],HF_HOME=cfg['HF_HOME'],EXTRA_SERVE_ARGS=cfg.get('EXTRA_SERVE_ARGS',''),HEALTH_WAIT_MIN='120',EXPERIMENT_CONFIG=str(root/'study_config.json'),PILOT_DEPLOYMENT_REPORT=str(root/'deployment_report.json'),PILOT_GENERATION_OVERRIDES=str(root/'generation_overrides.json'),PILOT_OUTPUT=str(root/'output'),PILOT_APPROVAL=str(root/'approval.json'),PILOT_SEED_START=str(seed))
    partition='gpu_h200_8' if alias=='gpt-oss-120b' else 'gpu_h100_4';qos='qos_gpu_h200' if alias=='gpt-oss-120b' else 'qos_gpu_h100'
    cmd=['sbatch','--parsable','--export=ALL','--partition='+partition,'--qos='+qos,'--gres=gpu:1','--cpus-per-task=4','--mem='+('128G' if alias=='gpt-oss-120b' else '96G'),'--time='+duration,'--job-name=agl_parallel_'+alias,str(repo/'cluster/sharanga_pilot.sbatch')]
    prepared.append((alias,root,seed,cmd,env))
state=subprocess.check_output(['squeue','-h','-j','381648','-o','%T'],text=True).strip()
assert state=='PENDING',state
subprocess.run(['scancel','381648'],check=True)
submitted=[]
for alias,root,seed,cmd,env in prepared:
    jid=subprocess.check_output(cmd,cwd=repo,env=env,text=True).strip().split(';')[0]
    submitted.append({'model':alias,'job_id':jid,'output':str(root/'output'),'seed_start':seed,'dependency':None,'parser_version':'final_decision_v2'})
    (bundle/'recovery_submissions.json').write_text(json.dumps(submitted,indent=2)+'\n')
    print(alias,jid,flush=True)
