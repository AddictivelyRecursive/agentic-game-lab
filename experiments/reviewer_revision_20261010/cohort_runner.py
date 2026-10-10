"""One registered prospective episode with complete request/response telemetry.

No scheduler submission. Refuses collection without a dated internal archive receipt
bound to the cohort manifest and protocol lock. Run --validate-only without GPUs.
"""
import argparse
import csv
from datetime import datetime,timezone
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import subprocess
import time
from types import SimpleNamespace
from urllib.parse import urlsplit
from cohort_protocol import messages_for,ROOT,HERE
from frozen_parser_v2 import extract_decision_json as v2,validate_decision
from parser_v3 import extract_decision_json as v3
from game_engine.env.types import EnvConfig,PayoffConfig,DriftConfig,StreakConfig,ObservationConfig,AgentMeta
from game_engine.env.simulator import GameSimulator
from game_engine.agents.llm_wrapper import LLMWrapperAgent

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def check_registration(receipt,manifest,lock):
    r=json.loads(Path(receipt).read_text())
    if r.get('record_type')!='internal_prospective_archive':
        raise ValueError('Dated internal prospective archive receipt required')
    if r.get('manifest_sha256')!=digest(manifest) or r.get('protocol_lock_sha256')!=digest(lock):
        raise ValueError('Archive receipt does not match exact manifest and protocol lock')
    stamp=datetime.fromisoformat(r['archived_at_utc'].replace('Z','+00:00'))
    if stamp.tzinfo is None or stamp.utcoffset().total_seconds()!=0 or stamp>datetime.now(timezone.utc):
        raise ValueError('Archive timestamp must be a past UTC timestamp')
    return r

def append(path,record):
    with path.open('a',encoding='utf-8') as f:
        f.write(json.dumps(record,ensure_ascii=False)+'\n'); f.flush(); os.fsync(f.fileno())

class Agent:
    def __init__(self,cfg,row,client,tokenizer,output,context,max_tokens):
        self.cfg=cfg; self.row=row; self.client=client; self.tokenizer=tokenizer; self.output=output; self.context=context; self.max_tokens=max_tokens
    def reset(self,seed): pass
    def act(self,obs):
        turn=LLMWrapperAgent._build_turn(SimpleNamespace(env_cfg=self.cfg),obs)
        repair=None; calls=[]; chosen=None; started=time.monotonic()
        for attempt in range(4):
            messages=messages_for(turn,self.row['arm'],repair)
            identity=f"{self.row['episode_id']}:{obs.agent_id}:{obs.t}:{attempt}"
            seed=int.from_bytes(hashlib.sha256(identity.encode()).digest()[:4],'little')%2147483647
            payload=dict(model=self.row['model'],messages=messages,temperature=.7,top_p=1.,max_tokens=self.max_tokens,seed=seed)
            record=dict(utc=datetime.now(timezone.utc).isoformat(),episode_id=self.row['episode_id'],round=obs.t,seat=obs.agent_id,attempt=attempt,stage='decision' if attempt==0 else 'repair',request=payload,raw_response=None,error=None,finish_reason=None)
            append(self.output/'requests_started.jsonl',dict(utc=record['utc'],episode_id=record['episode_id'],round=obs.t,seat=obs.agent_id,attempt=attempt,request=payload))
            tick=time.monotonic(); content=''
            try:
                token_response=self.client.post(self.base_url.removesuffix('/v1')+'/tokenize',json=dict(model=self.row['model'],messages=messages,add_generation_prompt=True),timeout=60,allow_redirects=False)
                token_response.raise_for_status()
                count=int(token_response.json()['count']); record['prompt_tokens_server']=count
                if count+self.max_tokens>self.context: raise ValueError('Context bound exceeded; no prompt truncation performed')
                response=self.client.post(self.base_url+'/chat/completions',json=payload,timeout=600,allow_redirects=False)
                record['http_status']=response.status_code
                try: record['raw_response']=response.json()
                except ValueError: record['raw_response_text']=response.text
                if 300<=response.status_code<400: raise ValueError('Redirect forbidden')
                response.raise_for_status()
                choice=record['raw_response']['choices'][0]; record['finish_reason']=choice.get('finish_reason'); record['usage']=record['raw_response'].get('usage')
                content=choice['message'].get('content')
                if not isinstance(content,str): raise ValueError('No final content; reasoning field is never substituted')
            except Exception as exc:
                if record.get('prompt_tokens_server') is None or 'Context bound exceeded' in str(exc):
                    record['error']=str(exc); append(self.output/'requests.jsonl',record)
                    raise RuntimeError('Tokenizer/context infrastructure check failed; episode incomplete') from exc
                record['error']=str(exc)
            record['elapsed_s']=time.monotonic()-tick
            # Persist the full transport response before extracting a decision.
            append(self.output/'requests.jsonl',record)
            try:
                if record['error']: raise ValueError(record['error'])
                parser=v3 if self.row['parser']=='v3_literal_whitespace_only_20261011' else v2
                result=validate_decision(dict(decision=parser(content),M=self.cfg.M))
                if not result['is_valid']: raise ValueError(result['validation_error'])
                chosen=result['decision']['a']; valid=True; error=None
            except (ValueError,TypeError) as exc:
                valid=False; error=str(exc); repair=dict(error=error,output=content if isinstance(content,str) else '')
            calls.append(dict(attempt=attempt,valid=valid,error=error,finish_reason=record['finish_reason']))
            if valid: break
        fallback=chosen is None
        if fallback: chosen=4
        decision=dict(episode_id=self.row['episode_id'],round=obs.t,seat=obs.agent_id,action=chosen,first_valid=calls[0]['valid'],repair_used=len(calls)>1,repair_calls=len(calls)-1,fallback_used=fallback,calls=calls,elapsed_s=time.monotonic()-started)
        append(self.output/'decisions.jsonl',decision)
        return chosen,AgentMeta(agent_name=self.row['model'],parse_ok=calls[0]['valid'],action_ok=True,fallback_used=fallback,fallback_reason='four_failed_attempts->a4' if fallback else None,extra={'prospective':True,'protocol_parser':self.row['parser'],'decision':decision})

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--manifest',type=Path,required=True); ap.add_argument('--episode-id',required=True); ap.add_argument('--validate-only',action='store_true'); ap.add_argument('--registration',type=Path); ap.add_argument('--lock',type=Path); ap.add_argument('--base-url'); ap.add_argument('--model-path'); ap.add_argument('--output',type=Path); ap.add_argument('--max-tokens',type=int); ap.add_argument('--context',type=int,default=16384); ap.add_argument('--resume-partial',action='store_true'); a=ap.parse_args()
    rows=list(csv.DictReader(a.manifest.open(encoding='utf-8'))); selected=[r for r in rows if r['episode_id']==a.episode_id]
    if len(selected)!=1: raise ValueError('Expected exactly one prespecified episode')
    row=selected[0]
    if row['parser'] not in ['final_decision_v2','v3_literal_whitespace_only_20261011']: raise ValueError('Protocol unresolved; no enrollment')
    frozen=row['dynamics']=='frozen'
    cfg=EnvConfig(N=int(row['N']),M=5,T=int(row['T']),p_perception=0.,seed=int(row['seed']),payoff=PayoffConfig(B0=12.,C=8.,K=0.,B_min=12. if frozen else 9.,B_max=12. if frozen else 15.),drift=DriftConfig(window_w=8,eta=0. if frozen else .35,r_star=.55),streak=StreakConfig(theta=.5,lam=0. if frozen else .25,tau=4.),obs=ObservationConfig(history_k=10,stats_window=10))
    if a.validate_only: print(json.dumps(dict(episode=row,config=asdict(cfg),collection=False))); return
    if not a.registration or not a.lock: raise ValueError('Internal prospective archive and protocol lock required before collection')
    receipt=check_registration(a.registration,a.manifest,a.lock)
    lock=json.loads(a.lock.read_text())
    for name,sha in lock['files'].items():
        if digest(ROOT/name)!=sha: raise ValueError('Protocol source changed: '+name)
    settings=lock['model_settings'].get(row['model'])
    if not settings or settings.get('status')!='LOCKED': raise ValueError('Model settings not locked')
    if row['parser']!=settings['parser']: raise ValueError('Parser differs from locked model settings')
    for package,version in lock['inference_software'].items():
        if importlib.metadata.version(package)!=version: raise ValueError('Inference software changed: '+package)
    if a.max_tokens!=settings['max_tokens'] or a.context!=settings['context']: raise ValueError('Generation allowance differs from registered settings')
    if not a.model_path or Path(a.model_path).name!=settings['snapshot']: raise ValueError('Snapshot differs from registered settings')
    for name,sha in settings['artifact_sha256'].items():
        if digest(Path(a.model_path)/name)!=sha: raise ValueError('Model/tokenizer artifact changed: '+name)
    for name,size in settings['weight_sizes'].items():
        if (Path(a.model_path)/name).stat().st_size!=size: raise ValueError('Weight shard missing or size changed: '+name)
    url=urlsplit(a.base_url or '')
    if url.scheme!='http' or url.hostname not in ['localhost','127.0.0.1'] or url.path!='/v1' or url.query or url.fragment or url.username or url.password: raise ValueError('Loopback endpoint required')
    if not os.getenv('SLURM_JOB_ID'): raise ValueError('Collection requires a Slurm allocation')
    if not a.output: raise ValueError('Output required')
    if a.output.exists():
        if (a.output/'completed.json').exists(): raise ValueError('Completed episode exists; never rerun or select among completed outcomes')
        if not a.resume_partial: raise ValueError('Partial output exists; explicit --resume-partial required')
        resolved=a.output.resolve(); allowed=ROOT.resolve()
        if not resolved.is_relative_to(allowed) or resolved==allowed: raise ValueError('Partial archive must stay within this isolated study workspace')
        previous=json.loads((resolved/'manifest.json').read_text())
        if previous['episode']['episode_id']!=row['episode_id'] or previous['protocol_lock_sha256']!=digest(a.lock): raise ValueError('Partial episode or protocol differs')
        job=str(previous['slurm_job_id'])
        if not job.isdigit() or job==os.environ['SLURM_JOB_ID']: raise ValueError('Cannot archive this job or an unidentified prior allocation')
        r=subprocess.run(['sacct','-X','-j',job,'--format=State','--noheader','--parsable2'],capture_output=True,text=True,check=True)
        states=[s.split('|')[0].split()[0] for s in r.stdout.splitlines() if s.strip()]
        if not states or any(s not in ['CANCELLED','FAILED','TIMEOUT','COMPLETED','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED'] for s in states): raise ValueError('Prior allocation is not verified terminal')
        index=1
        while resolved.with_name(resolved.name+f'.partial_attempt_{index:03d}').exists(): index+=1
        archived=resolved.with_name(resolved.name+f'.partial_attempt_{index:03d}')
        resolved.rename(archived)
        print('Archived interrupted episode attempt without deleting records:',archived,flush=True)
    a.output.mkdir(parents=True,exist_ok=False)
    import requests
    client=requests.Session(); client.trust_env=False
    models=client.get(a.base_url+'/models',timeout=30,allow_redirects=False); models.raise_for_status()
    if row['model'] not in [m['id'] for m in models.json()['data']]: raise ValueError('Endpoint does not serve the registered model name')
    tokenizer=None  # Actual serving tokenizer is used through /tokenize, including Mistral mode.
    partials=[p.name for p in a.output.parent.glob(a.output.name+'.partial_attempt_*')]
    (a.output/'manifest.json').write_text(json.dumps(dict(episode=row,registration=receipt,protocol_lock_sha256=digest(a.lock),model_path=a.model_path,slurm_job_id=os.environ['SLURM_JOB_ID'],prior_partial_attempts=partials,allocation_environment={k:os.getenv(k) for k in ['SLURM_JOB_PARTITION','SLURM_JOB_GPUS','CUDA_VISIBLE_DEVICES','SLURM_CPUS_PER_TASK']},inference_software={p:importlib.metadata.version(p) for p in lock['inference_software']}),indent=2))
    agents=[Agent(cfg,row,client,tokenizer,a.output,a.context,a.max_tokens) for _ in range(cfg.N)]
    for ag in agents: ag.base_url=a.base_url
    started=time.monotonic()
    try:
        result=GameSimulator(cfg).run_episode(agents)
        if frozen and any(r.B_eff!=12. for r in result.logs): raise ValueError('Frozen benefit assertion failed')
        for r in result.logs:
            if any(m.fallback_used and act!=4 for m,act in zip(r.agent_meta,r.true_actions)): raise ValueError('Unexpected environment fallback')
            append(a.output/'rounds.jsonl',asdict(r))
        (a.output/'completed.json').write_text(json.dumps(dict(status='completed',rounds=len(result.logs),elapsed_s=time.monotonic()-started,cooperation=sum(sum(r.true_coop)/cfg.N for r in result.logs)/cfg.T),indent=2))
    except Exception as exc:
        (a.output/'incomplete.json').write_text(json.dumps(dict(status='incomplete',error=str(exc),elapsed_s=time.monotonic()-started),indent=2)); raise

if __name__=='__main__': main()
