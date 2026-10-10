"""Descriptive Stage 0 summaries and cost sheet; no episode-level inference."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from diagnose_saved import table
HERE=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--responses',type=Path,nargs='+',required=True); ap.add_argument('--bank',type=Path,default=HERE/'request_bank.jsonl'); a=ap.parse_args()
    bank={r['id']:r for r in map(json.loads,a.bank.read_text(encoding='utf-8').splitlines())}; rows=[]
    expected_sha=hashlib.sha256(a.bank.read_bytes()).hexdigest()
    for path in a.responses:
        # Refuse to mix the compact-JSON pilot with exact-message calibration.
        job=json.loads((path.parent.parent/'job.json').read_text())
        files=job.get('sources',{})
        if files.get('request_bank.jsonl')!=expected_sha: raise ValueError('Nonmatching/unknown diagnostic bank: '+str(path))
        for record in map(json.loads,path.read_text(encoding='utf-8').splitlines()):
            if record['state_id'] not in bank: raise ValueError('Unknown state')
            usage=record.get('usage') or {}
            rows.append(dict(model=record['model'],state_id=record['state_id'],max_tokens=record['max_tokens'],first_valid=record['first_valid'],length=record['finish_reason']=='length',error=record['error'] is not None,elapsed_s=record['elapsed_s'],completion_tokens=usage.get('completion_tokens',np.nan),prompt_tokens=usage.get('prompt_tokens',np.nan)))
    dd=pd.DataFrame(rows)
    if dd.duplicated(['model','state_id','max_tokens']).any(): raise ValueError('Duplicate responses; no outcome selection allowed')
    result=[]; costs=[]
    for (model,budget),g in dd.groupby(['model','max_tokens']):
        expected=12 if model=='qwen3-8b' else 4
        complete=len(g)==expected
        mean=g.elapsed_s.mean()
        result.append(dict(model=model,max_tokens=budget,expected=expected,observed=len(g),complete=complete,first_valid_pct=100*g.first_valid.mean(),length_pct=100*g.length.mean(),errors=int(g.error.sum()),mean_latency_s=mean,mean_completion_tokens=g.completion_tokens.mean(),max_completion_tokens=g.completion_tokens.max(),note='Descriptive saved-state diagnostic, not independent episodes'))
        for n in [2,3,4,5]:
            costs.append(dict(model=model,max_tokens=budget,N=n,expected_first_attempt_episode_minutes=mean*30*n/60,mean_request_latency_s=mean,full_12_seeds_per_8_cells_first_attempt_gpu_hours=mean*10080/3600,complete_calibration=complete,n_requests=len(g),note='Provisional extrapolation, excludes repairs/loading and new-history effects; never a guaranteed budget'))
    table(pd.DataFrame(result),'00_exact_calibration')
    table(pd.DataFrame(costs),'00_calibration_cost_sheet')
    print(pd.DataFrame(result).to_string(index=False))

if __name__=='__main__': main()
