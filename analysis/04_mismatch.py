"""POST-HOC / EXPLORATORY: rule-based explanation audit; no fabricated labels."""
import argparse
import hashlib
import uuid
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportion_confint
from detector_rules import detect
from common import *

DEFAULT_EXPORT=ROOT/'results/analysis/main_h5_coding_20261010'

def prevalence(dd,keys,name):
    rows=[]
    for key,sub in dd.groupby(keys,observed=True):
        if not isinstance(key,tuple): key=(key,)
        for scope in ['all_nonempty','valid_unrepaired']:
            s=sub if scope=='all_nonempty' else sub[sub.first_valid&~sub.repair&~sub.fallback]
            n=len(s); k=int(s.any_rule.sum()); low,high=proportion_confint(k,n,method='wilson') if n else (np.nan,np.nan)
            row=dict(zip(keys,key)); row.update(scope=scope,decisions=n,episodes=s.episode_id.nunique(),flagged=k,percent=100*k/n if n else np.nan,wilson_low=100*low,wilson_high=100*high)
            for r in ['R1','R2','R3','R4']: row[r+'_count']=int(s[r].sum())
            # Seat-weighted prevalence bootstrap retains episode clusters; N cells stratified.
            random=rng('audit '+str(key)+' '+scope); totals=np.zeros(5000); flagged=np.zeros(5000)
            for _,cell in sub.groupby('N'):
                all_ids=cell.episode_id.unique(); c=s[s.N==cell.N.iloc[0]].groupby('episode_id').any_rule.agg(['sum','size']).reindex(all_ids,fill_value=0)
                ix=random.integers(0,len(c),(5000,len(c))); x=c.to_numpy(); draw=x[ix].sum(axis=1); flagged+=draw[:,0]; totals+=draw[:,1]
            valid=totals>0; boot=100*flagged[valid]/totals[valid]
            row['cluster_low'],row['cluster_high']=np.quantile(boot,[.025,.975]) if len(boot) else (np.nan,np.nan)
            row['bootstrap_invalid_draws']=int((~valid).sum()); rows.append(row)
    result=pd.DataFrame(rows); table(result,name); return result

def main():
    require_gate(); ap=argparse.ArgumentParser(); ap.add_argument('--input',default=DEFAULT_INPUT); ap.add_argument('--export-dir',type=Path,default=DEFAULT_EXPORT); a=ap.parse_args(); _,all_decisions,_=load(a.input)
    rules_hash=hashlib.sha256((HERE/'detector_rules.py').read_bytes()).hexdigest()
    # Freeze rules before aggregate outcome inspection.
    (HERE/'results/04_rules_manifest.json').write_text(json.dumps(dict(rules_sha256=rules_hash,seed=SEED,version=1,note='Frozen before prevalence; only known examples inspected'),indent=2))
    export=pd.read_csv(a.export_dir/'coding_sheet.csv',keep_default_na=False); key=json.loads((a.export_dir/'private_condition_key.json').read_text())
    assert key['source_sha256']==SOURCE_HASH
    # Original exporter hashes JSON-serialized values, not raw UTF-8 text.
    def original_digest(value):
        return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    assert original_digest(key['rows'])==key['sample_sha256']
    kd=pd.DataFrame(key['rows']); assert kd.row_id.is_unique and export.row_id.is_unique
    joined=export.merge(kd,on='row_id',validate='one_to_one'); assert len(joined)==19740
    scope=all_decisions[all_decisions.N.isin([2,5])].copy()
    linked=joined.merge(scope,on=['episode_id','round','seat'],validate='one_to_one',suffixes=('_export',''))
    assert len(linked)==len(scope)==19740
    assert (linked.reasoning_text==linked.text).all()
    assert (linked.model_export.map(MODELS)==linked.model).all()
    assert (linked.reasoning_mode==linked['mode']).all() and (linked.N_export==linked.N).all() and (linked.seed_export==linked.seed).all()
    assert linked.apply(lambda r: original_digest(r.text)==r.text_sha256,axis=1).all()
    empty=scope.text.str.strip().eq(''); assert empty.sum()==274
    table(scope.assign(empty=empty).groupby(['model','mode']).agg(decisions=('text','size'),empty_explanations=('empty','sum')).reset_index(),'04_empty_explanations')
    dd=scope[~empty].copy(); detected=pd.DataFrame([detect(r.text,r.action) for r in dd.itertuples()],index=dd.index)
    dd=pd.concat([dd,detected],axis=1); dd['bucket']=pd.cut(dd['round'],bins=[0,5,25,30],labels=['1-5','6-25','26-30'])
    known=[]
    for model,n,seed,t,seat in [('Gemma',2,1001,15,0),('Llama',2,1001,1,0)]:
        s=dd[(dd.model==model)&(dd['mode']=='structured')&(dd.N==n)&(dd.seed==seed)&(dd['round']==t)&(dd.seat==seat)]
        assert len(s)==1; r=s.iloc[0]; known.append(dict(model=model,mode='structured',N=n,seed=seed,round=t,seat=seat,action=r.action,flagged=bool(r.any_rule),rules=','.join(k for k in ['R1','R2','R3','R4'] if r[k])))
        assert r.any_rule,f'Known example not caught: {model}; stop before prevalence'
    table(pd.DataFrame(known),'04_known_examples')
    modelmode=prevalence(dd,['model','mode'],'04_mismatch_by_model_mode'); roundrates=prevalence(dd,['model','mode','bucket'],'04_mismatch_by_round')
    detail=dd[['episode_id','model','mode','N','seed','round','seat','action','R1','R2','R3','R4','any_rule','evidence']].copy(); detail.evidence=detail.evidence.map(json.dumps); table(detail,'04_decision_flags')
    random=rng('validation sample'); picked=[]; private=[]
    for flag in [True,False]:
        cells=[dd[(dd.model==m)&(dd.any_rule==flag)] for m in MODELS.values()]; alloc=np.zeros(6,int)
        for _ in range(75):
            candidates=[i for i,c in enumerate(cells) if alloc[i]<len(c)]
            if not candidates: raise ValueError('Insufficient decisions to satisfy 75 cases in flag stratum')
            i=min(candidates,key=lambda j:(alloc[j],j)); alloc[i]+=1
        for i,c in enumerate(cells):
            for ix in random.choice(c.index,size=alloc[i],replace=False):
                r=dd.loc[ix]; ident=uuid.UUID(bytes=random.bytes(16)).hex
                picked.append(dict(decision_id=ident,explanation_text=r.text,chosen_action=int(r.action),action_map='a=0 full cooperation (1.00); a=1 (0.75); a=2 (0.50); a=3 (0.25); a=4 full defection (0.00)',human_label=''))
                private.append(dict(decision_id=ident,model=r.model,mode=r['mode'],episode_id=r.episode_id,round=int(r['round']),seat=int(r.seat),detector_flag=flag,stratum_population=len(c),stratum_sample=int(alloc[i]),sampling_weight=len(c)/alloc[i]))
    sample=pd.DataFrame(picked).iloc[random.permutation(150)].reset_index(drop=True); assert sample.human_label.eq('').all()
    sample.to_csv(HERE/'validation/validation_sample.csv',index=False)
    (HERE/'results/latex/04_validation_sample.tex').write_text('% POST-HOC / EXPLORATORY; blinded labeling artifact\n'+sample.to_latex(index=False,escape=True,column_format='lp{8cm}ll l'),encoding='utf-8')
    pd.DataFrame(private).to_csv(HERE/'validation/private_validation_key.csv',index=False)
    table(pd.DataFrame(private),'04_private_validation_key')
    table(dd.groupby(['model','any_rule']).size().reset_index(name='population'),'04_sampling_strata')
    for mode in MODES:
        s=modelmode[(modelmode['mode']==mode)&(modelmode.scope=='all_nonempty')].set_index('model').reindex(MODELS.values())
        fig,ax=plt.subplots(figsize=(8,3)); ax.bar(s.index,s.percent); ax.errorbar(s.index,s.percent,yerr=[s.percent-s.wilson_low,s.wilson_high-s.percent],fmt='none',color='black',label='Wilson (descriptive)'); ax.set_ylabel('Detectable inconsistencies (%)'); ax.set_title('POST-HOC / EXPLORATORY — '+mode); fig.tight_layout(); fig.savefig(HERE/'figures'/f'04_prevalence_{mode}.pdf'); plt.close(fig)
    for (m,mode),s in roundrates[roundrates.scope=='all_nonempty'].groupby(['model','mode']):
        fig,ax=plt.subplots(figsize=(5,3)); ax.plot(s.bucket.astype(str),s.percent,'o-'); ax.set_ylabel('Flagged (%)'); ax.set_title(f'POST-HOC / EXPLORATORY — {m} {mode}'); fig.tight_layout(); fig.savefig(HERE/'figures'/f'04_round_{m}_{mode}.pdf'); plt.close(fig)
    total=int(dd.any_rule.sum()); text=f'Frozen four-rule detector (SHA256 {rules_hash}) audited {len(dd):,} nonempty explanations from {len(scope):,} N=2/5 decisions. The 274 empty explanations were excluded and reported separately. Flags: {total:,}/{len(dd):,} ({100*total/len(dd):.2f}%). Both known examples were caught, as documented in 04_known_examples.csv. Export identity, explanation text hashes and canonical decision linkage all verified.\n\n'
    text+=modelmode[['model','mode','scope','decisions','flagged','percent','wilson_low','wilson_high','cluster_low','cluster_high']].to_string(index=False)+'\n\n'
    text+='Verdict: these are **rates of detectable inconsistencies under a rule-based detector**, not misunderstanding. Prevalence is exploratory and limited to N=2/5. Explicit local assertions improve specificity but miss paraphrases and complicated semantics; no aggregate outcome tuning was performed. R1 may flag an incorrect action-map assertion even if it describes prior play. Wilson intervals are requested descriptive binomial intervals and do not account for dependence; episode-stratified bootstrap intervals accompany them. Repairs and fallback exclusions yield a conditional, potentially selected subset.\n\n'
    text+='Validation: 150 blinded decisions, 75 flagged and 75 unflagged, stratified across models, exported with blank labels. The private key records unequal sampling fractions. No labels or precision/recall results were fabricated. score_validation.py requires real supplied labels; weighted recall corrects the case-control sampling ratio, with approximate effective-sample Wilson intervals. Human validation is pending. Optional LLM judge skipped: no API access or explicit judge authorization was supplied.'
    section('Task 4 — explanation mismatch audit',text); print(modelmode.to_string(index=False)); print('Total flagged',total,'/',len(dd))

if __name__=='__main__': main()
