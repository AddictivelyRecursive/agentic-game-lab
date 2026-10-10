"""One-time bounded initial cohort submission, two dependency lanes, persistent receipts."""
import hashlib,json,os,subprocess
from pathlib import Path
ROOT=Path('/home/jagatsesh/f20220604_ts/agentic-game-lab/results/sharanga/internal_cohort_20261011')
HERE=ROOT/'experiments/reviewer_revision_20261010'
receipt=HERE/'registration/initial_collection_submissions.json'
if receipt.exists(): raise SystemExit('Receipt exists: do not duplicate jobs')
lock=json.loads((HERE/'registration/protocol_lock_stage3.json').read_text())
for name,sha in lock['files'].items():
 if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha: raise ValueError('Source mismatch '+name)
chunks=json.loads((HERE/'registration/chunks.json').read_text())['chunks']
selected=[c for c in chunks if c['cohort_id']=='rr_stage3_fourarm_v2' and c['block']==1 and c['model'] in ['mistral-small-3.1-24b-instruct','gemma-3-12b-it']]
assert len(selected)==3,[(c['chunk_id'],len(c['episode_ids'])) for c in selected]
(ROOT/'job_logs').mkdir(exist_ok=True)
lanes={m:'388864:388865' for m in ['mistral-small-3.1-24b-instruct','gemma-3-12b-it']}
records=[]
for c in selected:
 m=c['model']; setting=lock['model_settings'][m]
 env=dict(os.environ,BUNDLE=str(ROOT),MODEL=m,MODEL_PATH=setting['snapshot_path'],CHUNK=c['chunk_id'])
 args=['sbatch','--parsable','--nodes=1','--ntasks=1','--gres=gpu:1','--cpus-per-task=4','--mem=96G','--time='+str(c['wall_minutes']),'--partition=gpu_h100_4','--qos=qos_gpu_h100','--dependency=afterany:'+lanes[m],'--job-name=rr_'+m.split('-')[0]+'_b01','--output='+str(ROOT/'job_logs/%j.out'),'--export=ALL',str(HERE/'cohort.sbatch')]
 result=subprocess.run(args,env=env,text=True,capture_output=True,check=True)
 job=result.stdout.strip().split(';')[0]
 if not job.isdigit(): raise ValueError('Unexpected receipt '+result.stdout)
 records.append(dict(job_id=job,chunk=c,dependency=args[10] if False else lanes[m],protocol_lock_sha256=hashlib.sha256((HERE/'registration/protocol_lock_stage3.json').read_bytes()).hexdigest(),externally_registered=False))
 receipt.write_text(json.dumps(records,indent=2))
 lanes[m]=job
 print(job,c['chunk_id'],flush=True)
