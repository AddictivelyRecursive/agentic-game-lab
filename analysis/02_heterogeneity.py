"""POST-HOC / EXPLORATORY: heterogeneity sensitivity and resampling."""
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import *

def boot_effects(ep,B=5000):
    random=rng('heterogeneity within cell'); ds=[]; ses=[]; dfs=[]
    for model in MODELS.values():
        estimate=np.zeros(B); variance=np.zeros(B); denom=np.zeros(B)
        for mode in MODES:
            for n in range(2,6):
                x=ep[(ep.model==model)&(ep['mode']==mode)&(ep.N==n)].c.to_numpy()*100
                draws=x[random.integers(0,len(x),(B,len(x)))]; w=(1 if mode=='free_form' else -1)*.6*(n-3.5)
                estimate+=w*draws.mean(axis=1); v=w*w*draws.var(axis=1,ddof=1)/len(x); variance+=v; denom+=v*v/(len(x)-1)
        ds.append(estimate); ses.append(np.sqrt(variance)); dfs.append(np.divide(variance**2,denom,out=np.full(B,np.nan),where=denom>0))
    return np.array(ds).T,np.array(ses).T,np.array(dfs).T

def q_array(d,se):
    valid=np.isfinite(se).all(axis=1)&(se>0).all(axis=1); q=np.full(len(d),np.nan); i2=q.copy()
    w=1/se[valid]**2; pool=(w*d[valid]).sum(axis=1)/w.sum(axis=1)
    q[valid]=(w*(d[valid]-pool[:,None])**2).sum(axis=1); i2[valid]=100*np.maximum(0,1-5/q[valid])
    return q,i2,valid

def main():
    require_gate(); ap=argparse.ArgumentParser(); ap.add_argument('--input',default=DEFAULT_INPUT); a=ap.parse_args(); ep,_,_=load(a.input)
    effects=contrast(ep); observed=heterogeneity(effects); leave=[]
    for model in MODELS.values():
        h=heterogeneity(effects[effects.model!=model]); leave.append(dict(excluded=model,df=4,**h,significant=bool(h['p']<.05)))
    loo=pd.DataFrame(leave); table(loo,'02_leave_one_out')
    rows=[]; random=rng('parametric Q')
    for typ in ['normal','t']:
        noise=random.normal(size=(20000,6)) if typ=='normal' else random.standard_t(np.maximum(2,effects.df.to_numpy()),size=(20000,6))
        d=observed['pooled']+noise*effects.se.to_numpy(); q,_,_=q_array(d,np.tile(effects.se.to_numpy(),(20000,1)))
        p=(np.sum(q>=observed['Q'])+1)/20001
        rows.append(dict(method='parametric_'+typ,draws=20000,valid_draws=20000,invalid_draws=0,observed_Q=observed['Q'],p=p,mc_se=np.sqrt(p*(1-p)/20000),note='Fixed observed SE; t uses SE as scale and df>=2, not unit-variance t'))
    d,se,df=boot_effects(ep); q,i2,valid=q_array(d,se)
    p=2*stats.t.sf(np.divide(abs(d),se,out=np.full_like(d,np.nan),where=se>0),df)
    holm=np.full(len(d),np.nan)
    eligible=np.isfinite(p).all(axis=1)
    order=np.argsort(p[eligible],axis=1)
    sorted_p=np.take_along_axis(p[eligible],order,axis=1)
    adjusted=np.minimum(1,np.maximum.accumulate(sorted_p*np.arange(6,0,-1),axis=1))
    holm[eligible]=adjusted[np.arange(eligible.sum()),np.argmax(order==0,axis=1)]
    # Verify vectorized Holm against statsmodels on a representative valid draw.
    if eligible.any():
        b=np.flatnonzero(eligible)[0]
        assert np.isclose(holm[b],multipletests(p[b],method='holm')[1][0])
    for metric,values in [('Q',q),('I2',i2)]:
        finite=values[np.isfinite(values)]; low,high=np.quantile(finite,[.025,.975]) if len(finite) else (np.nan,np.nan)
        rows.append(dict(method='episode_bootstrap_'+metric,draws=5000,valid_draws=len(finite),invalid_draws=5000-len(finite),low=low,high=high,note='Percentile CI CONDITIONAL on all six positive SEs; unconditional interval not estimable if invalid draws occur'))
    eligible=np.isfinite(holm); sig=int(np.sum(holm[eligible]<.05)); validn=int(eligible.sum())
    rows.append(dict(method='episode_bootstrap_Llama_Holm',draws=5000,valid_draws=validn,invalid_draws=5000-validn,fraction=sig/validn,low=sig/5000,high=(sig+5000-validn)/5000,note='fraction conditional on estimable six-model Holm; low/high bound unconditional fraction by unresolved draws'))
    table(pd.DataFrame(rows),'02_Q_resampling')
    table(pd.DataFrame(dict(draw=np.arange(5000),Q=q,I2=i2,Llama_holm=holm,all_positive_se=valid)),'02_bootstrap_draws')
    endpoint=contrast(ep,endpoint=True); joined=effects.merge(endpoint,on='model',suffixes=('_fourpoint','_endpoint')); table(joined,'02_endpoint_vs_fourpoint')
    fig,ax=plt.subplots(figsize=(8,4)); y=np.arange(6)
    ax.errorbar(effects.D,y,xerr=[effects.D-effects.low,effects.high-effects.D],fmt='o',capsize=3)
    pooled=observed['pooled']; half=stats.norm.ppf(.975)*observed['pooled_se']; ax.errorbar(pooled,6,xerr=half,fmt='D',color='black',capsize=4)
    ax.set_yticks(range(7),list(MODELS.values())+['Pooled (descriptive)']); ax.axvline(0,color='gray',ls=':'); ax.set_xlabel('Four-point interaction D (pp), 95% CI'); ax.invert_yaxis(); ax.set_title('POST-HOC / EXPLORATORY'); fig.tight_layout(); fig.savefig(HERE/'figures/02_forest.pdf'); plt.close(fig)
    llama=loo[loo.excluded=='Llama'].iloc[0]; normal=rows[0]; tnull=rows[1]
    text=f'Inverse-variance leave-one-model-out analysis: **Heterogeneity robust to leaving out Llama? {"yes" if llama.p<.05 else "no"}.** Excluding Llama gives Q(4)={llama.Q:.5f}, p={llama.p:.5f}, I²={llama.I2:.2f}%. '+('The more defensible claim is that one model differs; omnibus significance depends on Llama.' if llama.p>=.05 else 'The remaining five-model subset retains omnibus significance.')+'\n\n'
    text+=loo[['excluded','Q','p','I2','significant']].to_string(index=False)+'\n\n'
    text+=f'Parametric null simulated p: Normal {normal["p"]:.5f} (MC SE {normal["mc_se"]:.5f}), t-scale {tnull["p"]:.5f} (MC SE {tnull["mc_se"]:.5f}); 20,000 draws each. The t version uses scale=observed SE and Welch df floored at 2, so it deliberately has heavier variance. Observed Q is {observed["Q"]:.8f} (rounded 18.23).\n\n'
    text+=pd.DataFrame(rows).to_string(index=False)+'\n\n'
    text+=f'**Bootstrap limitation:** {5000-int(valid.sum())}/5000 resamples have at least one zero/nonfinite model SE. No floors or silent deletions were used. Listed finite Q/I² intervals are conditional sensitivity summaries, not unconditional 95% intervals. Six-model Holm is estimable in {validn}/5000 draws; conditional Llama significance fraction {sig/validn:.4f}, unconditional bounds [{sig/5000:.4f}, {(sig+5000-validn)/5000:.4f}]. Per-draw results are saved.\n\n'
    text+='Literal endpoints and four-point contrasts have these estimates (pp):\n'+joined[['model','D_fourpoint','low_fourpoint','high_fourpoint','D_endpoint','low_endpoint','high_endpoint']].to_string(index=False)+'\n\n'
    text+='Verdict: distinguish model sensitivity of the omnibus test from an assertion that all models differ. Endpoint and four-point definitions need not agree in magnitude because intermediate N cells contribute only to the latter. The pooled forest marker uses a normal fixed-weight CI and is descriptive; six estimated model SEs and sparse GPT-OSS cells limit chi-square and bootstrap calibration.'
    section('Task 2 — heterogeneity robustness',text); print(loo.to_string(index=False)); print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__': main()
