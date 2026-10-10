"""POST-HOC / EXPLORATORY: saved-data loaders and episode-level statistics only."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
SEED = 20261010
MODELS = dict(zip(['llama-3.1-8b-instruct','qwen3-8b','gemma-3-12b-it',
                  'mistral-small-3.1-24b-instruct','deepseek-r1-distill-qwen-14b','gpt-oss-120b'],
                 ['Llama','Qwen','Gemma','Mistral','DeepSeek','GPT-OSS']))
COUNTS = dict(zip(MODELS.values(), [14,4,8,3,16,2]))
MODES = ['structured','free_form']
SOURCE_HASH = '149ac53f8a19286b429d1de83939ae91028845b0b35a18e1dc8dd5cc0468b778'
DEFAULT_INPUT = ROOT/'results/server_copies/final_main_20261010/main_complete_20261010/main.jsonl'
for folder in ['results/latex','figures','validation']:
    (HERE/folder).mkdir(parents=True, exist_ok=True)

def rng(name):
    return np.random.default_rng(np.random.SeedSequence([SEED, int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], 'little')]))

def table(df, name):
    df.to_csv(HERE/'results'/f'{name}.csv', index=False)
    formatters={c:(lambda x: f'{x:.4f}') for c in df.columns if c in ['p','holm','p_fourpoint','p_endpoint','holm_fourpoint','holm_endpoint','mc_se']}
    tex = df.to_latex(index=False, escape=True, float_format=lambda x: f'{x:.2f}', formatters=formatters, na_rep='not estimable',longtable=len(df)>40)
    (HERE/'results/latex'/f'{name}.tex').write_text('% POST-HOC / EXPLORATORY\n'+tex, encoding='utf-8')

def load(path=DEFAULT_INPUT):
    from frozen_parser_v2 import extract_decision_json, validate_decision
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SOURCE_HASH:
        raise ValueError(f'Unapproved source hash: {digest}')
    episodes=[]; decisions=[]; rounds=[]; record_types={}; parsers=set(); stages=set()
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            if not line.strip(): continue
            row=json.loads(line)
            typ=row.get('record_type'); record_types[typ]=record_types.get(typ,0)+1
            if 'decisions' not in row: continue
            for key in ['episode_id','model','reasoning_mode','N','seed','status','rounds','decisions']:
                if key not in row: raise ValueError(f'Missing episode field {key}')
            assert row['model'] in MODELS and row['reasoning_mode'] in MODES
            n=row['N']; assert n in range(2,6)
            ident=dict(episode_id=row['episode_id'],model=MODELS[row['model']],mode=row['reasoning_mode'],N=n,seed=row['seed'])
            rs={r['t']:r for r in row['rounds']}; assert len(rs)==30 and set(rs)==set(range(1,31))
            values=[]
            for t,r in sorted(rs.items()):
                actions=r['true_actions']; assert len(actions)==n
                assert all(type(a) is int and 0<=a<=4 for a in actions)
                cs=[1-a/4 for a in actions]; assert cs==r['true_coop']
                values.append(np.mean(cs)); rounds.append(dict(**ident,round=t,c=np.mean(cs)))
            seen=set()
            for d in row['decisions']:
                for key in ['round','seat','repair_used','repair_calls','fallback_used','reasoning_text','first_parse_ok','model_calls','aborted']:
                    if key not in d: raise ValueError(f'Missing decision field {key}')
                t,s=d['round'],d['seat']; assert (t,s) not in seen and t in rs and s in range(n); seen.add((t,s))
                a=rs[t]['true_actions'][s]
                assert type(d['repair_used']) is bool and type(d['fallback_used']) is bool
                assert not d['aborted'] and 0<=d['repair_calls']<=3
                assert not d['fallback_used'] or a==4
                calls=d['model_calls']; assert calls
                for call in calls:
                    if 'output' not in call and not call.get('error'):
                        raise ValueError('Missing saved call output without a logged failure; STOP')
                    parsers.add(str(call.get('json_extraction_version'))); stages.add(str(call.get('stage')))
                try:
                    obj=extract_decision_json(calls[0].get('output',''))
                    parse=True; valid=validate_decision(dict(decision=obj,M=5))['is_valid']
                except (ValueError,TypeError): parse=valid=False
                decisions.append(dict(**ident,round=t,seat=s,action=a,c=1-a/4,
                                      repair=d['repair_used'],repair_calls=d['repair_calls'],fallback=d['fallback_used'],
                                      text=d['reasoning_text'],first_parse=parse,first_valid=valid,legacy_parse=d['first_parse_ok']))
            assert len(seen)==30*n
            assert np.isclose(np.mean(values),row['cooperation'],atol=1e-12)
            episodes.append(dict(**ident,c=np.mean(values),early=np.mean(values[:5]),late=np.mean(values[-5:]),
                                 opening=values[0],late_zero=float(all(v==0 for v in values[-5:])),
                                 slope=np.polyfit(np.arange(1,31),values,1)[0],status=row['status']))
    ep=pd.DataFrame(episodes); dd=pd.DataFrame(decisions); rr=pd.DataFrame(rounds)
    assert len(ep)==376 and len(dd)==39480 and ep.episode_id.is_unique
    assert not ep.duplicated(['model','mode','N','seed']).any()
    cells=ep.groupby(['model','mode','N']).size(); assert len(cells)==48
    for (m,mode,n),count in cells.items(): assert count==COUNTS[m]
    assert set(ep.status)=={'completed'},set(ep.status)
    rates=dd.groupby('episode_id')[['repair','fallback']].mean()
    ep=ep.merge(rates,left_on='episode_id',right_index=True,validate='one_to_one')
    meta=dict(input=str(path),sha256=digest,record_types=record_types,parser_versions=sorted(parsers),call_stages=sorted(stages),episodes=len(ep),decisions=len(dd),seed=SEED,frozen_parser_sha256=hashlib.sha256((HERE/'frozen_parser_v2.py').read_bytes()).hexdigest())
    (HERE/'results/source_manifest.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    return ep,dd,rr

def means(ep, cols=('c','early','late')):
    return ep.groupby(['model','mode','N'])[list(cols)].mean().groupby(['model','mode']).mean().reset_index()

def contrast(ep, endpoint=False):
    out=[]
    for model in MODELS.values():
        sub=ep[ep.model==model]; terms=[]; estimate=0.; denom=0.; ok=True
        for mode in MODES:
            for n in ([2,5] if endpoint else range(2,6)):
                x=sub[(sub['mode']==mode)&(sub.N==n)].c.to_numpy()*100
                if len(x)<2: ok=False; continue
                w=(1 if mode=='free_form' else -1)*((1 if n==5 else -1) if endpoint else .6*(n-3.5))
                estimate+=w*x.mean(); v=w*w*x.var(ddof=1)/len(x); terms.append(v); denom+=v*v/(len(x)-1)
        variance=sum(terms)
        row=dict(model=model,D=estimate if ok else np.nan,se=np.nan,df=np.nan,low=np.nan,high=np.nan,p=np.nan,status='not estimable')
        if ok and variance>0 and denom>0:
            se=np.sqrt(variance); df=variance**2/denom; half=stats.t.ppf(.975,df)*se
            row.update(se=se,df=df,low=estimate-half,high=estimate+half,p=2*stats.t.sf(abs(estimate/se),df),status='estimated')
        out.append(row)
    out=pd.DataFrame(out); out['holm']=np.nan
    if out.p.notna().all(): out['holm']=multipletests(out.p,method='holm')[1]
    return out

def heterogeneity(effects):
    d=effects.D.to_numpy(); se=effects.se.to_numpy(); k=len(d)
    if not np.isfinite(se).all() or (se<=0).any(): return dict(Q=np.nan,p=np.nan,I2=np.nan,pooled=np.nan,pooled_se=np.nan)
    w=1/se**2; pooled=np.sum(w*d)/sum(w); q=np.sum(w*(d-pooled)**2)
    return dict(Q=q,p=stats.chi2.sf(q,k-1),I2=100*max(0,(q-k+1)/q) if q else 0.,pooled=pooled,pooled_se=1/np.sqrt(sum(w)))

def require_gate():
    p=HERE/'results/00_gate.json'
    if not p.exists() or not json.loads(p.read_text())['passed']: raise RuntimeError('Task 0 gate has not passed. STOP.')

def section(title, text):
    p=HERE/'RESULTS.md'
    content=p.read_text(encoding='utf-8') if p.exists() else '**POST-HOC / EXPLORATORY**\n\n'
    marker=f'## {title}\n'
    replacement=marker+'\n'+text+'\n\n'
    if marker in content:
        start=content.index(marker)
        end=content.find('\n## ',start+len(marker))
        suffix=content[end+1:] if end>=0 else ''
        content=content[:start]+replacement+suffix
    else:
        content+=replacement
    p.write_text(content,encoding='utf-8')
