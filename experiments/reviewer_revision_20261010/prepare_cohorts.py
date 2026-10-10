"""Prospective design preparation and saved-data precision planning; no inference."""
import hashlib
import json
from pathlib import Path
import random
import sys
import numpy as np
import pandas as pd
from scipy import stats
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'analysis'))
from common import SEED,MODELS
from diagnose_saved import table

def main():
    old=pd.read_csv(ROOT/'Thesis/data/main_episode_means_20261010.csv')
    precision=[]
    for model,g in old.groupby('model'):
        cells=g.groupby(['mode','N']).cooperation.var(ddof=1)*10000
        for n in [12,20,40]:
            for margin in [1.,1.5,2.]:
                terms=np.array([(.6*(N-3.5))**2*v*margin/n for (mode,N),v in cells.items()])
                var=terms.sum(); df=var**2/np.sum(terms**2/(n-1)); half=stats.t.ppf(.975,df)*np.sqrt(var)
                precision.append(dict(model=model,seeds_per_cell=n,variance_multiplier=margin,se_pp=np.sqrt(var),welch_df=df,half_width_pp=half,target_half_width_pp=10,meets_target=bool(half<=10),basis='Plug-in original cell variances; not guaranteed precision'))
            # Cooperation in [0,100] has population variance <=2500. Use the
            # lowest possible Welch df n-1 for a conservative t-based benchmark.
            se=np.sqrt(3.6*2500/n)
            precision.append(dict(model=model,seeds_per_cell=n,variance_multiplier=np.nan,se_pp=se,welch_df=n-1,half_width_pp=stats.t.ppf(.975,n-1)*se,target_half_width_pp=10,meets_target=False,basis='Bounded-outcome variance benchmark; t approximation, not a finite-sample guarantee'))
    table(pd.DataFrame(precision),'planning_precision')
    timing=old.groupby('model').elapsed_s.agg(['count','mean','median','max']).reset_index()
    timing['historical_gpu_hours_for_96_episodes_excluding_load']=96*timing['mean']/3600
    timing['note']='Historical sequential episode runtime; changed protocols and loading can change cost substantially'
    table(timing,'planning_historical_costs')
    # Seeds are globally disjoint across cohorts, arms and N cells. Generation
    # seeds derive from episode ID, seat, round and attempt in the runner.
    seed=200001; cohorts={}
    def add(cid,models,arms,ns,count,dynamics,stage,parser):
        nonlocal seed
        rows=[]
        for model in models:
            for n in ns:
                for arm in arms:
                    for block in range(count):
                        eid=f'{cid}__{model}__{arm}__n{n}__seed{seed}'
                        rows.append(dict(cohort_id=cid,stage=stage,model=model,N=n,arm=arm,seed=seed,block=block+1,episode_id=eid,T=30,dynamics=dynamics,parser=parser.get(model,'final_decision_v2'),status='NOT_REGISTERED_NOT_SUBMITTED'))
                        seed+=1
        # Fixed randomized order within each balanced block and model; this
        # preserves whole-episode boundaries and model-homogeneous chunks.
        random.Random(SEED+stage).shuffle(rows)
        rows.sort(key=lambda r:(r['block'],r['model']))
        cohorts[cid]=rows
    llama='llama-3.1-8b-instruct'; mistral='mistral-small-3.1-24b-instruct'; gemma='gemma-3-12b-it'; gpt='gpt-oss-120b'
    add('rr_stage1_reliability_v2',['qwen3-8b','deepseek-r1-distill-qwen-14b'],['S','F'],[2,5],2,'dynamic',1,{})
    add('rr_stage2_independent_v2',[mistral,gpt],['S','F'],[2,3,4,5],12,'dynamic',2,{gpt:'PENDING_INSTRUMENTED_DIAGNOSTIC'})
    add('rr_stage3_fourarm_v2',[llama,gemma,mistral],['F','S','EV-canonical','EV-sorted'],[2,3,4,5],12,'dynamic',3,{llama:'v3_literal_whitespace_only_20261011'})
    add('rr_stage4_floor_smoke_v2',[llama,mistral],['S','F'],[2,5],2,'frozen',4,{llama:'v3_literal_whitespace_only_20261011'})
    add('rr_stage4_frozen_v2',[llama,mistral],['S','F'],[2,3,4,5],12,'frozen',4,{llama:'v3_literal_whitespace_only_20261011'})
    dest=HERE/'registration'; dest.mkdir(exist_ok=True)
    allrows=[]
    for cid,rows in cohorts.items():
        path=dest/(cid+'_episodes.csv'); pd.DataFrame(rows).to_csv(path,index=False)
        allrows+=rows
    assert len({r['seed'] for r in allrows})==len(allrows)
    assert not any(1001<=r['seed']<=1016 for r in allrows)
    manifest=dict(seed=SEED,prospective=True,registered=False,external_registration_required=True,cohorts={cid:dict(episodes=len(rows),manifest_sha256=hashlib.sha256((dest/(cid+'_episodes.csv')).read_bytes()).hexdigest()) for cid,rows in cohorts.items()},floor_rule='Smoke F mean cooperation <5% at BOTH N=2 and N=5; calculated across two episodes per N, before inspecting slopes. S is descriptive only. Smoke seeds are not reused in the primary frozen cohort.',stage4_dynamic_control='Stage3 S/F rows only; exact model, parser, prompt and runtime hashes must match. Otherwise no sharing.',unresolved=['Stage1 token budgets await exact-message calibration','Stage2 GPT-OSS protocol awaits instrumented diagnostic','Stage4 larger-model allocation awaits deployment and smoke checks','Full-cohort GPU budget and chunk sizing await measured timings'])
    (dest/'cohort_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2))

if __name__=='__main__': main()
