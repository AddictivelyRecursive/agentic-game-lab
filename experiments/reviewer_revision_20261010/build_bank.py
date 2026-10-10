"""Prepare a metadata-selected saved-state diagnostic bank; no inference."""
import hashlib
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from src.AI_Agent.agent.pipeline import prepare_context,build_payload

def main():
    shards=ROOT/'results/server_copies/snapshot_20261009_2103/results/sharanga/main_short_chunks_20261007/shards'
    prompts=ROOT/'src/AI_Agent/prompts'; rows=[]
    for model in ['qwen3-8b','deepseek-r1-distill-qwen-14b']:
        for mode in ['structured','free_form']:
            for n in [2,5]:
                eid=f'{model}__{mode}__n{n}__seed1001'
                paths=list(shards.glob(f'{model}__chunk*/{model}/output/episodes/{eid}/agents/p0/agent_traces.jsonl'))
                if not paths:
                    paths=list((ROOT/'results/server_copies/reviewer_calibration_sources_20261010').glob(eid+'.jsonl'))
                if len(paths)!=1: raise ValueError(f'Expected one saved trace for {eid}: {paths}')
                with paths[0].open(encoding='utf-8') as stream:
                    states={x['round']:x for x in map(json.loads,stream) if x['round'] in [1,15,30]}
                for t in [1,15,30]:
                    saved=states[t]; state=prepare_context(dict(raw_input=saved['raw_input']))
                    payload=build_payload(state,dict(reasoning_mode=mode,memory_mode='history_only',predict_opponents=False))
                    if mode=='structured': assert state['eu_ranking']==saved['eu_ranking']
                    policy='decision_policy_structured.txt' if mode=='structured' else 'decision_policy_system.txt'
                    system=(prompts/'base_system.txt').read_text()+'\n'+(prompts/policy).read_text()
                    user=json.dumps(payload,ensure_ascii=False,indent=2)
                    uid=f'{eid}__t{t}__p0'
                    rows.append(dict(id=uid,model=model,mode=mode,N=n,round=t,seat=0,episode_seed=1001,system=system,user=user,source_trace=str(paths[0].relative_to(ROOT)),source_trace_sha256=hashlib.sha256(paths[0].read_bytes()).hexdigest(),request_seed=int.from_bytes(hashlib.sha256(uid.encode()).digest()[:4],'little')%2147483647))
    target=HERE/'request_bank.jsonl'; target.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    assert len(rows)==24
    print('Prepared',len(rows),'bank states; v2 selects 12 Qwen + 4 DeepSeek states for 32 requests. SHA256',hashlib.sha256(target.read_bytes()).hexdigest())

if __name__=='__main__': main()
