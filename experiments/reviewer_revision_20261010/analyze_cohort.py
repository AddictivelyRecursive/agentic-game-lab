"""Prospective Stage 2/3 analysis; infer only after every planned episode completes."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

def estimate(cells):
    # Each input is (weight, vector of independent EPISODE outcomes in percent).
    if any(len(x)<2 for w,x in cells): return dict(estimate_pp=np.nan,se_pp=np.nan,df=np.nan,low_pp=np.nan,high_pp=np.nan,p=np.nan,status='not estimable: <2 episodes in a cell')
    mean=sum(w*np.mean(x) for w,x in cells)
    terms=[w*w*np.var(x,ddof=1)/len(x) for w,x in cells]
    var=sum(terms); denominator=sum(v*v/(len(x)-1) for v,(w,x) in zip(terms,cells))
    if var<=0 or denominator<=0: return dict(estimate_pp=mean,se_pp=0.,df=np.nan,low_pp=np.nan,high_pp=np.nan,p=np.nan,status='zero estimated variance; t inference not estimable')
    se=np.sqrt(var); df=var**2/denominator; half=stats.t.ppf(.975,df)*se
    return dict(estimate_pp=mean,se_pp=se,df=df,low_pp=mean-half,high_pp=mean+half,p=2*stats.t.sf(abs(mean/se),df),status='estimated')

def contrast(ep,model,a,b,slope=False):
    cells=[]
    for arm,sign in [(a,1),(b,-1)]:
        for n in [2,3,4,5]:
            x=ep[(ep.model==model)&(ep.arm==arm)&(ep.N==n)].c.to_numpy()*100
            cells.append((sign*(.6*(n-3.5) if slope else .25),x))
    return estimate(cells)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--manifest',type=Path,required=True); ap.add_argument('--episodes-root',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    manifest=pd.read_csv(a.manifest); expected=set(manifest.episode_id); data=[]; incomplete=[]
    if len(expected)!=len(manifest): raise ValueError('Duplicate prespecified episode IDs')
    if set(manifest.stage)=={3} and (len(manifest)!=576 or set(manifest.model)!={'llama-3.1-8b-instruct','gemma-3-12b-it','mistral-small-3.1-24b-instruct'}): raise ValueError('Stage 3 registered roster/allocation changed')
    if set(manifest.stage)=={2} and (len(manifest)!=192 or set(manifest.model)!={'mistral-small-3.1-24b-instruct','gpt-oss-120b'}): raise ValueError('Stage 2 registered roster/allocation changed')
    if set(manifest.stage) in [{2},{3}] and (set(manifest.N)!={2,3,4,5} or not (manifest.groupby(['model','N','arm']).size()==12).all() or not (manifest['T']==30).all()): raise ValueError('Registered cell allocation or horizon changed')
    for _,row in manifest.iterrows():
        folder=a.episodes_root/row.episode_id
        if not (folder/'completed.json').exists(): incomplete.append(row.episode_id); continue
        recorded=json.loads((folder/'manifest.json').read_text())['episode']
        if any(str(recorded[k])!=str(row[k]) for k in ['episode_id','model','N','arm','parser']): raise ValueError('Episode metadata differs from prespecified row')
        completion=json.loads((folder/'completed.json').read_text()); logs=[json.loads(s) for s in (folder/'rounds.jsonl').read_text().splitlines()]
        if len(logs)!=30 or {r['t'] for r in logs}!=set(range(1,31)): raise ValueError('Incomplete/duplicate rounds in claimed completion')
        logs=sorted(logs,key=lambda r:r['t']); cs=[]
        for r in logs:
            if len(r['true_actions'])!=row.N or any(type(x) is not int or not 0<=x<=4 for x in r['true_actions']): raise ValueError('Invalid actions')
            c=np.mean([1-x/4 for x in r['true_actions']]); assert np.isclose(c,np.mean(r['true_coop']))
            cs.append(c)
        if not np.isclose(np.mean(cs),completion['cooperation']): raise ValueError('Episode outcome mismatch')
        data.append(dict(episode_id=row.episode_id,model=row.model,arm=row.arm,N=row.N,c=np.mean(cs),early=np.mean(cs[:5]),late=np.mean(cs[-5:]),late_minus_early=np.mean(cs[-5:])-np.mean(cs[:5]),late_zero=bool(all(c==0 for c in cs[-5:]))))
    a.output.mkdir(parents=True,exist_ok=True); ep=pd.DataFrame(data); ep.to_csv(a.output/'episode_outcomes.csv',index=False)
    (a.output/'completion_status.json').write_text(json.dumps(dict(planned=len(expected),completed=len(ep),incomplete=incomplete,inference_allowed=not incomplete),indent=2))
    if incomplete: print('Incomplete cohort: descriptive outcomes retained; no significance tests computed.'); return
    stage=set(manifest.stage); result=[]
    if stage=={3}:
        contrasts=[('EV-canonical','F'),('EV-sorted','EV-canonical'),('S','EV-sorted')]
        for model in sorted(ep.model.unique()):
            for arm,b in contrasts:
                for slope in [False,True]:
                    result.append(dict(model=model,contrast=arm+' minus '+b,estimand='D' if slope else 'level_gap',family='stage3_interactions_secondary_9' if slope else 'stage3_levels_primary_9',**contrast(ep,model,arm,b,slope)))
        llama='llama-3.1-8b-instruct'; r=contrast(ep,llama,'F','S',True)
        r['p']=stats.t.sf(r['estimate_pp']/r['se_pp'],r['df']) if r['status']=='estimated' else np.nan
        result.append(dict(model=llama,contrast='F minus S',estimand='D_positive_directional_replication',family='single_directional_replication',**r))
    elif stage=={2}:
        for model in sorted(ep.model.unique()):
            for slope in [False,True]: result.append(dict(model=model,contrast='F minus S',estimand='D' if slope else 'level_gap',family='stage2_primary_6',**contrast(ep,model,'F','S',slope)))
            cells=[(.25,ep[(ep.model==model)&(ep.arm=='F')&(ep.N==n)].late_minus_early.to_numpy()*100) for n in [2,3,4,5]]
            result.append(dict(model=model,contrast='F late minus early',estimand='persistence',family='stage2_primary_6',**estimate(cells)))
    else: raise ValueError('This locked analysis supports only Stage 2 and Stage 3')
    out=pd.DataFrame(result); out['holm']=np.nan
    for family,indices in out.groupby('family').groups.items():
        if family=='single_directional_replication': continue
        p=out.loc[indices,'p']; corrected=multipletests(p.fillna(1),method='holm')[1]
        out.loc[indices,'holm']=np.where(p.notna(),corrected,np.nan)
    out.to_csv(a.output/'contrasts.csv',index=False)
    (a.output/'latex').mkdir(exist_ok=True)
    (a.output/'latex/contrasts.tex').write_text('% PROSPECTIVE FOLLOW-UP; episode-level inference\n'+out.to_latex(index=False,escape=True,float_format=lambda x:f'{x:.2f}'),encoding='utf-8')
    # Temporal summaries are descriptive and episode-resampled within N cells.
    rng=np.random.default_rng(20261010); temporal=[]
    for (model,arm),g in ep.groupby(['model','arm']):
        for metric in ['late_minus_early','late_zero']:
            cells=[cell[metric].to_numpy(dtype=float) for n,cell in g.groupby('N')]
            draws=np.mean([x[rng.integers(0,len(x),(5000,len(x)))].mean(axis=1) for x in cells],axis=0)*100
            lo,hi=np.quantile(draws,[.025,.975]); temporal.append(dict(model=model,arm=arm,metric=metric,mean_pct=100*np.mean([x.mean() for x in cells]),low_pct=lo,high_pct=hi,status='descriptive; episode bootstrap'))
    td=pd.DataFrame(temporal); td.to_csv(a.output/'temporal.csv',index=False)
    (a.output/'latex/temporal.tex').write_text('% PROSPECTIVE FOLLOW-UP; descriptive\n'+td.to_latex(index=False,float_format=lambda x:f'{x:.2f}'),encoding='utf-8')

if __name__=='__main__': main()
