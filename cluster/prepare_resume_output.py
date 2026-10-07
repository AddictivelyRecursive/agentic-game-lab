"""Archive interrupted attempts before resuming unchanged completed records."""
import hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
output=Path(os.environ['PILOT_OUTPUT']).resolve()
model=os.environ['MODEL']
jobid=os.environ.get('SLURM_JOB_ID','preflight')
owner=output/'.resume_owner.json'
if owner.exists():
    previous=json.loads(owner.read_text())['job_id']
    if previous != jobid and previous != 'preflight':
        active=dict(line.split(maxsplit=1) for line in subprocess.check_output(['squeue','-h','-o','%i %T'],text=True).splitlines() if line.strip())
        state=active.get(previous,'')
        if state in ('RUNNING','COMPLETING','CONFIGURING'):
            raise RuntimeError('Another recovery job is active: '+previous)
phase=os.environ.get('STUDY_PHASE','pilot')
if phase not in ('pilot','main'):raise ValueError('Invalid study phase')
raw=output/(phase+'.jsonl')
if not raw.exists():
    owner.write_text(json.dumps({'job_id':jobid,'model':model}))
    print('Fresh '+phase+' output; nothing to recover')
    sys.exit(0)
original=raw.read_bytes()
lines=original.splitlines(keepends=True)
rows=[];tail=b''
for i,line in enumerate(lines):
    try:r=json.loads(line)
    except (ValueError,UnicodeDecodeError):
        if i != len(lines)-1:raise RuntimeError('Corrupt non-tail JSONL record')
        tail=line;continue
    if i:rows.append(r)
if any(r.get('status')!='completed' for r in rows):
    raise RuntimeError('Recorded failed episodes need separate review; never silently replace them')
completed={r['episode_id'] for r in rows}
archive=output/'interrupted_attempts'/f'{jobid}_{time.time_ns()}'
actions=[]
if tail:
    archive.mkdir(parents=True)
    shutil.copy2(raw,archive/(phase+'_before_tail_recovery.jsonl'))
    raw.write_bytes(b''.join(lines[:-1]))
    actions.append('Archived incomplete trailing JSONL bytes')
for folder in (output/'episodes').glob(model+'__*'):
    if folder.name in completed:continue
    assert folder.resolve().parent == (output/'episodes').resolve()
    archive.mkdir(parents=True,exist_ok=True)
    shutil.move(str(folder),str(archive/folder.name))
    actions.append('Archived interrupted episode '+folder.name)
lock=output/'.execution.lock'
if lock.exists():
    archive.mkdir(parents=True,exist_ok=True)
    shutil.move(str(lock),str(archive/'execution.lock'))
    actions.append('Archived stale execution lock')
receipt={'job_id':jobid,'model':model,'completed_episodes_retained':sum(r['model']==model for r in rows),'completed_records_sha256':hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest(),'actions':actions}
if actions:
    (archive/'recovery.json').write_text(json.dumps(receipt,indent=2))
owner.write_text(json.dumps({'job_id':jobid,'model':model}))
print(json.dumps(receipt),flush=True)
