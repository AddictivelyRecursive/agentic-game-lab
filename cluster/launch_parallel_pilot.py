"""Prepare isolated parser-v2 shards, then replace authorized pending jobs."""
import json, os, shutil, subprocess, sys
from pathlib import Path
base = Path('/home/jagatsesh/f20220604_ts/agentic-game-lab')
bundle = base / 'results/sharanga/parallel_parser_v2_20261005'
bundle.mkdir(exist_ok=False)
repo = bundle / 'code'
subprocess.run(['git','clone','--quiet','--no-hardlinks',str(base),str(repo)],check=True)
for folder in ('src','experiments','cluster'):
    shutil.copytree(base/folder,repo/folder,dirs_exist_ok=True,ignore=shutil.ignore_patterns('results','outputs','__pycache__'))
for path in base.glob('*.py'):
    shutil.copy2(path,repo/path.name)
shutil.copy2('/tmp/agl_pipeline_v2.py',repo/'src/AI_Agent/agent/pipeline.py')
shutil.copy2('/tmp/agl_test_json_extraction.py',repo/'tests/test_json_extraction.py')
lib=repo/'cluster/lib.sh'
lib.write_text(lib.read_text().replace('--approval "${PILOT_APPROVAL:?}" --execute-model "$MODEL"', '--approval "${PILOT_APPROVAL:?}" --execute-model "$MODEL" --seed-start "${PILOT_SEED_START:-101}"'))
subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p','test_json_extraction.py','-q'],cwd=repo,check=True)
subprocess.run(['bash','-n',str(lib),str(repo/'cluster/sharanga_pilot.sbatch')],check=True)
sys.path.insert(0,str(repo))
from experiments.study_common import digest, code_fingerprint
old=base/'results/sharanga/pilot30_temp07_20261003'
prior=json.loads((old/'output/run_plan.json').read_text())
specs=[('gpt-oss-120b',101,'20:00:00'),('llama-3.1-8b-instruct',106,'03:00:00'),('qwen3-8b',106,'30:00:00')]
os.environ['AGL_SUBMIT_GIT_COMMIT'] = subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
os.environ['AGL_SUBMIT_GIT_STATUS'] = subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=repo,text=True)
prepared=[]
for alias,seed,duration in specs:
    root=bundle/alias;root.mkdir()
    for name in ('study_config.json','deployment_report.json','generation_overrides.json'):
        shutil.copy2(old/name,root/name)
    subprocess.run([sys.executable,str(repo/'run_pilot.py'),'--models',*prior['models'],'--config',str(root/'study_config.json'),'--deployment-report',str(root/'deployment_report.json'),'--generation-overrides',str(root/'generation_overrides.json'),'--output',str(root/'output'),'--seed-start',str(seed),'--dry-run'],cwd=repo,check=True,stdout=subprocess.DEVNULL)
    plan=json.loads((root/'output/run_plan.json').read_text())
    assert all(c['generation']['temperature']==0.7 and c['generation']['top_p']==1.0 for c in plan['cells'])
    (root/'approval.json').write_text(json.dumps({'plan_sha256':digest(plan),'confirmed_in_conversation':True,'approval_reference':'2026-10-05 user explicitly authorized JSON extraction fix, separate outputs, cancellation of sequential GPT-OSS/Llama/Qwen jobs and parallel replacements'},indent=2))
    record=prior['deployment']['models'][alias];job=json.loads((Path(record['evidence_path'])/'job.json').read_text());cfg=job['environment']
    env=dict(os.environ,REPO_ROOT=str(repo),MODEL=alias,MODEL_REPO=cfg['MODEL_REPO'],MODEL_PATH=job['vllm_command'][2],SERVED_NAME=alias,TP_SIZE='1',PORT=cfg['PORT'],MAX_MODEL_LEN=cfg['MAX_MODEL_LEN'],GPU_MEM_UTIL=cfg['GPU_MEM_UTIL'],CONDA_ENV=cfg['CONDA_ENV'],HF_HOME=cfg['HF_HOME'],EXTRA_SERVE_ARGS=cfg.get('EXTRA_SERVE_ARGS',''),HEALTH_WAIT_MIN='120',EXPERIMENT_CONFIG=str(root/'study_config.json'),PILOT_DEPLOYMENT_REPORT=str(root/'deployment_report.json'),PILOT_GENERATION_OVERRIDES=str(root/'generation_overrides.json'),PILOT_OUTPUT=str(root/'output'),PILOT_APPROVAL=str(root/'approval.json'),PILOT_SEED_START=str(seed))
    partition='gpu_h200_8' if alias=='gpt-oss-120b' else 'gpu_h100_4';qos='qos_gpu_h200' if alias=='gpt-oss-120b' else 'qos_gpu_h100'
    cmd=['sbatch','--parsable','--export=ALL','--partition='+partition,'--qos='+qos,'--gres=gpu:1','--cpus-per-task=4','--mem='+('128G' if alias=='gpt-oss-120b' else '96G'),'--time='+duration,'--job-name=agl_parallel_'+alias,str(repo/'cluster/sharanga_pilot.sbatch')]
    prepared.append((alias,root,seed,cmd,env))
# Confirm all replacements ready before cancelling only the three requested jobs.
for jid in ('378646','379198','379199'):
    state=subprocess.check_output(['squeue','-h','-j',jid,'-o','%T'],text=True).strip()
    if state != 'PENDING':raise RuntimeError('Expected pending job '+jid+', found '+state)
(bundle/'provenance.json').write_text(json.dumps({'parser_version':'final_decision_v2','code_fingerprint':code_fingerprint(),'replaced_jobs':['378646','379198','379199'],'active_deepseek_unchanged':True},indent=2))
subprocess.run(['scancel','378646','379198','379199'],check=True)
submitted=[]
for alias,root,seed,cmd,env in prepared:
    jid=subprocess.check_output(cmd,cwd=repo,env=env,text=True).strip().split(';')[0]
    submitted.append({'model':alias,'job_id':jid,'output':str(root/'output'),'seed_start':seed,'dependency':None,'parser_version':'final_decision_v2'})
    (bundle/'submissions.json').write_text(json.dumps(submitted,indent=2)+'\n')
    print(alias,jid,flush=True)
