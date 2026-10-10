"""Prepare whole-episode model-homogeneous jobs; no scheduler mutations."""
import json
from pathlib import Path
import pandas as pd
from diagnose_saved import table
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]

def main():
    old=pd.read_csv(ROOT/'Thesis/data/main_episode_means_20261010.csv').groupby('model').elapsed_s.mean().to_dict()
    rows=[]; chunks=[]
    for cid in ['rr_stage2_independent_v2','rr_stage3_fourarm_v2']:
        manifest=pd.read_csv(HERE/'registration'/(cid+'_episodes.csv'))
        for (model,block),g in manifest.groupby(['model','block'],sort=False):
            # 15-minute provisional loading overhead and 2x historical episode
            # runtime. These MUST be replaced by calibration timings where the
            # token budget/protocol changes. Never split an episode across jobs.
            seconds=old[model]*2; limit=2*3600; per_chunk=max(1,int((limit-15*60)//seconds))
            episode_ids=g.episode_id.tolist()
            for start in range(0,len(episode_ids),per_chunk):
                ids=episode_ids[start:start+per_chunk]; cid_chunk=f'{cid}__{model}__b{block:02d}__part{start//per_chunk+1:02d}'
                minutes=max(60,min(240,int((15*60+len(ids)*seconds+299)//300)*5))
                chunk=dict(chunk_id=cid_chunk,cohort_id=cid,model=model,block=int(block),episode_ids=ids,gpus=1,cpus=4,wall_minutes=minutes,estimated_loading_minutes=15,timing_margin=2.,timing_basis='Original-budget episode mean; provisional; never an enrollment authorization',status='AWAITING_EXTERNAL_REGISTRATION_AND_PROTOCOL_GATES')
                chunks.append(chunk)
                rows.append({k:v for k,v in chunk.items() if k!='episode_ids'}|{'episodes':len(ids)})
    table(pd.DataFrame(rows),'planning_chunks')
    (HERE/'registration/chunks.json').write_text(json.dumps(dict(study_gpu_ceiling=2,submission_allowed=False,execution_rule='Use two Slurm dependency lanes globally across stages, one GPU per lane. First cohort jobs wait for BOTH calibration jobs to finish. A 70B TP=2 window must drain both lanes. No polling-based duplicate submission.',chunks=chunks),indent=2))
    print('Prepared',len(chunks),'whole-episode chunks; none submitted')

if __name__=='__main__': main()
