"""Resume three interrupted pilot shards without changing inference or parser."""
import json,os,shutil,subprocess,sys
from pathlib import Path
base=Path('/home/jagatsesh/f20220604_ts/agentic-game-lab')
bundle=base/'results/sharanga/parallel_parser_v2_20261005'
recovery=base/'results/sharanga/resume_20261006'
recovery.mkdir(exist_ok=True)
assert not (recovery/"submissions.json").exists(), "Resume submissions already exist"
shutil.copy2('/tmp/agl_prepare_resume.py',recovery/'prepare_resume_output.py')
specs=[('deepseek-r1-distill-qwen-14b',base,base/'results/sharanga/pilot30_temp07_20261003',101,'378645','60:00:00'),('gpt-oss-120b',bundle/'code',bundle/'gpt-oss-120b',101,'381716','20:00:00'),('qwen3-8b',bundle/'code',bundle/'qwen3-8b',106,'381718','30:00:00')]
prepared=[]
validate="""import argparse,json,sys; from pathlib import Path; sys.path.insert(0,sys.argv[1]); from experiments.study_common import make_plan,digest; root=Path(sys.argv[2]); saved=json.loads((root/'output/run_plan.json').read_text()); args=argparse.Namespace(models=saved['models'],deployment_report=root/'deployment_report.json',config=root/'study_config.json',generation_overrides=root/'generation_overrides.json',seed_start=int(sys.argv[3])); current=make_plan(args,'pilot'); assert digest(current)==digest(saved),'Frozen plan/code mismatch'; print('Plan verified',digest(saved))"""
for alias,repo,root,seed,oldjid,duration in specs:
    active=dict(line.split(maxsplit=1) for line in subprocess.check_output(['squeue','-h','-o','%i %T'],text=True).splitlines() if line.strip())
    state=active.get(oldjid,'')
    if state:raise RuntimeError('Prior job is still scheduled: '+oldjid+' '+state)
    subprocess.run([sys.executable,'-c',validate,str(repo),str(root),str(seed)],cwd=repo,check=True)
    output=root/'output'
    records=[json.loads(l) for l in (output/'pilot.jsonl').read_text().splitlines()]
    plan=records[0]['plan']
    rows=records[1:]
    done=sum(r['model']==alias and r['status']=='completed' for r in rows)
    assert done<40
    previous=json.loads((repo/'results/sharanga'/('pilot_'+oldjid)/'job.json').read_text())
    cfg=previous['environment']
    env=dict(os.environ,REPO_ROOT=str(repo),MODEL=alias,MODEL_REPO=cfg['MODEL_REPO'],MODEL_PATH=previous['vllm_command'][2],SERVED_NAME=alias,TP_SIZE=cfg['TP_SIZE'],PORT=cfg['PORT'],MAX_MODEL_LEN=cfg['MAX_MODEL_LEN'],GPU_MEM_UTIL=cfg['GPU_MEM_UTIL'],CONDA_ENV=cfg['CONDA_ENV'],HF_HOME=cfg['HF_HOME'],EXTRA_SERVE_ARGS=cfg.get('EXTRA_SERVE_ARGS',''),HEALTH_WAIT_MIN='120',EXPERIMENT_CONFIG=str(root/'study_config.json'),PILOT_DEPLOYMENT_REPORT=str(root/'deployment_report.json'),PILOT_GENERATION_OVERRIDES=str(root/'generation_overrides.json'),PILOT_OUTPUT=str(output),PILOT_APPROVAL=str(root/'approval.json'),PILOT_SEED_START=str(seed))
    env['AGL_SUBMIT_GIT_COMMIT']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    env['AGL_SUBMIT_GIT_STATUS']=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=repo,text=True)
    env['RECOVERY_ROOT']=str(recovery)
    # Preflight archives only interrupted folders; completed records are immutable.
    subprocess.run([sys.executable,str(recovery/'prepare_resume_output.py')],env=env,check=True)
    lib=recovery/(alias+'_lib.sh')
    code=(repo/'cluster/lib.sh').read_text()
    if '--seed-start' not in code:
        code=code.replace('--approval "${PILOT_APPROVAL:?}" --execute-model "$MODEL"','--approval "${PILOT_APPROVAL:?}" --execute-model "$MODEL" --seed-start "${PILOT_SEED_START:-101}"')
    lib.write_text(code)
    env['RECOVERY_LIB']=str(lib)
    partition='gpu_h200_8' if alias=='gpt-oss-120b' else 'gpu_h100_4'
    qos='qos_gpu_h200' if alias=='gpt-oss-120b' else 'qos_gpu_h100'
    prepared.append((alias,repo,env,partition,qos,duration,done))
wrapper=recovery/'resume.sbatch'
wrapper.write_text("""#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
set -eo pipefail
source "$HOME/miniconda3/bin/activate" "$CONDA_ENV"
python "$RECOVERY_ROOT/prepare_resume_output.py"
export RUN_DIR="$RECOVERY_ROOT/logs/${MODEL}_${SLURM_JOB_ID}_r${SLURM_RESTART_COUNT:-0}_$(date +%s)"
source "$RECOVERY_LIB"
run_job pilot
""")
subprocess.run(['bash','-n',str(wrapper),*[str(recovery/(a+'_lib.sh')) for a,*_ in specs]],check=True)
submitted=[]
for alias,repo,env,partition,qos,duration,done in prepared:
    cmd=['sbatch','--parsable','--export=ALL','--partition='+partition,'--qos='+qos,'--gres=gpu:1','--cpus-per-task=4','--mem='+('128G' if alias=='gpt-oss-120b' else '96G'),'--time='+duration,'--job-name=agl_resume_'+alias,'--output='+str(recovery/'resume_%j.out'),str(wrapper)]
    jid=subprocess.check_output(cmd,cwd=repo,env=env,text=True).strip().split(';')[0]
    submitted.append({'model':alias,'job_id':jid,'previous_job':next(s[4] for s in specs if s[0]==alias),'completed_retained':done,'remaining':40-done,'dependency':None,'output':env['PILOT_OUTPUT']})
    (recovery/'submissions.json').write_text(json.dumps(submitted,indent=2)+'\n')
    print(alias,jid,'remaining',40-done,flush=True)
