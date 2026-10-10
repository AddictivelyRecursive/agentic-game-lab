"""Bounded prospective saved-state inference diagnostic, never an episode runner."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import time
from urllib.parse import urlsplit
from frozen_parser_v2 import extract_decision_json,validate_decision

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model',required=True,choices=['qwen3-8b','deepseek-r1-distill-qwen-14b']); ap.add_argument('--bank',type=Path,required=True); ap.add_argument('--output',type=Path); ap.add_argument('--base-url'); ap.add_argument('--model-path'); ap.add_argument('--validate-only',action='store_true'); a=ap.parse_args()
    states=[json.loads(x) for x in a.bank.read_text(encoding='utf-8').splitlines() if x.strip()]; states=[x for x in states if x['model']==a.model]; assert len(states)==12 and len({s['id'] for s in states})==12
    budgets=[2048 if a.model=='qwen3-8b' else 8192,16384]; tasks=[(s,b) for s in states for b in budgets]; random.Random(20261010).shuffle(tasks)
    if a.validate_only: print(json.dumps(dict(model=a.model,states=len(states),requests=len(tasks),budgets=budgets,max_context=32768,max_runtime_minutes=90))); return
    url=urlsplit(a.base_url or '')
    if url.scheme!='http' or url.hostname not in ['127.0.0.1','localhost'] or url.path!='/v1' or url.username or url.password: raise ValueError('Loopback endpoint required')
    if not a.output or not a.model_path: raise ValueError('output and cached model path required')
    a.output.mkdir(parents=True,exist_ok=False)
    import requests
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(a.model_path,local_files_only=True)
    session=requests.Session(); session.trust_env=False; deadline=time.monotonic()+90*60; rows=[]
    for s,budget in tasks:
        if time.monotonic()>=deadline: break
        messages=[dict(role='system',content=s['system']),dict(role='user',content=s['user'])]
        record=dict(state_id=s['id'],model=a.model,mode=s['mode'],N=s['N'],round=s['round'],max_tokens=budget,request_seed=s['request_seed'],first_parse=False,first_valid=False,finish_reason=None,error=None)
        started=time.monotonic(); response_json=None
        try:
            count=len(tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True)); record['estimated_prompt_tokens']=count
            if count+budget>32768: raise ValueError('Input plus completion exceeds context; no truncation performed')
            payload=dict(model=a.model,messages=messages,temperature=.7,top_p=1.,max_tokens=budget,seed=s['request_seed'])
            response=session.post(a.base_url+'/chat/completions',json=payload,timeout=min(600,max(1,deadline-time.monotonic())),allow_redirects=False)
            if 300<=response.status_code<400: raise ValueError('Redirect forbidden')
            response.raise_for_status(); response_json=response.json(); choice=response_json['choices'][0]; record['finish_reason']=choice.get('finish_reason'); record['usage']=response_json.get('usage')
            content=choice['message'].get('content')
            if not isinstance(content,str): raise ValueError('No final content returned; reasoning content is not substituted')
            try:
                decision=extract_decision_json(content); record['first_parse']=True; record['first_valid']=validate_decision(dict(decision=decision,M=5))['is_valid']; record['parsed_action']=decision.get('a')
            except (ValueError,TypeError) as e: record['parse_error']=str(e)
            record['unterminated_think']='<think>' in content and '</think>' not in content
        except Exception as e: record['error']=str(e)
        record['elapsed_s']=time.monotonic()-started; record['raw_response']=response_json
        with (a.output/'responses.jsonl').open('a',encoding='utf-8') as stream: stream.write(json.dumps(record,ensure_ascii=False)+'\n'); stream.flush(); os.fsync(stream.fileno())
        rows.append(record); print(s['id'],budget,record['finish_reason'],record['first_valid'],flush=True)
    summary=[]
    for budget in budgets:
        r=[x for x in rows if x['max_tokens']==budget]; n=len(r)
        summary.append(dict(model=a.model,max_tokens=budget,requested_states=12,completed_requests=n,errors=sum(x['error'] is not None for x in r),first_valid=sum(x['first_valid'] for x in r),length_finishes=sum(x['finish_reason']=='length' for x in r),elapsed_s=sum(x['elapsed_s'] for x in r),note='Technical diagnostic only; saved states are not independent episodes'))
    with (a.output/'summary.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=summary[0]); writer.writeheader(); writer.writerows(summary)
    (a.output/'summary.json').write_text(json.dumps(dict(complete=len(rows)==24,planned_requests=24,completed_requests=len(rows),bank_sha256=hashlib.sha256(a.bank.read_bytes()).hexdigest(),summary=summary),indent=2))
    if len(rows)!=24: raise SystemExit(3)

if __name__=='__main__': main()
