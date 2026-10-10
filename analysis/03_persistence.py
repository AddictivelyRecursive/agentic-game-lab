"""POST-HOC / EXPLORATORY: level × slope × persistence."""
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import *

def summarize(ep,rr,by_n=False):
    rows=[]; keys=['model','mode']+(['N'] if by_n else [])
    for key,sub in ep.groupby(keys):
        ident=dict(zip(keys,key)); random=rng('persistence '+str(key)); boots=[]; curves=[]
        for _,g in sub.groupby('N'):
            x=g[['early','late','slope','late_zero']].to_numpy(); ix=random.integers(0,len(g),(5000,len(g)))
            boots.append(x[ix].mean(axis=1))
            r=rr[rr.episode_id.isin(g.episode_id)].pivot(index='episode_id',columns='round',values='c').reindex(g.episode_id).to_numpy()
            curves.append((r,r[ix].mean(axis=1)))
        boot=np.mean(boots,axis=0); observed=sub.groupby('N')[['early','late','slope','late_zero']].mean().mean()
        estimates={'late_minus_early':((observed.late-observed.early)*100,(boot[:,1]-boot[:,0])*100),
                   'round_slope_pp':(observed.slope*100,boot[:,2]*100),'fraction_zero_late':(observed.late_zero,boot[:,3])}
        row=dict(**ident,episodes=len(sub))
        for name,(value,draws) in estimates.items():
            row[name]=value; row[name+'_low'],row[name+'_high']=np.quantile(draws,[.025,.975])
        mean_curve=np.mean([r.mean(axis=0) for r,_ in curves],axis=0); bc=np.mean([b for _,b in curves],axis=0)
        hits=np.flatnonzero(mean_curve<.5*mean_curve[0]); row['half_opening_round']=int(hits[0]+1) if len(hits) else 'never'; row['opening_cooperation_percent']=100*mean_curve[0]
        hit=bc<.5*bc[:,0,None]; finite=hit.any(axis=1); times=hit[finite].argmax(axis=1)+1
        row['crossing_boot_never_fraction']=1-finite.mean(); row['crossing_finite_low'],row['crossing_finite_high']=np.quantile(times,[.025,.975]) if len(times) else (np.nan,np.nan)
        row['crossing_CI_note']='finite-crossing quantiles only; never retained separately; strict threshold'
        rows.append(row)
    return pd.DataFrame(rows)

def main():
    require_gate(); ap=argparse.ArgumentParser(); ap.add_argument('--input',default=DEFAULT_INPUT); a=ap.parse_args(); ep,_,rr=load(a.input)
    persistence=summarize(ep,rr); supplement=summarize(ep,rr,True); table(persistence,'03_persistence'); table(supplement,'03_persistence_by_N')
    level=means(ep).pivot(index='model',columns='mode',values='c'); grid=contrast(ep)
    grid['level_gap_pp']=grid.model.map((level.free_form-level.structured)*100)
    free=persistence[persistence['mode']=='free_form'].drop(columns=['mode','episodes']); grid=grid.merge(free,on='model')
    grid['persistence_class']=np.where(grid.late_minus_early<0,'decays','persists'); table(grid,'03_summary_grid')
    correlations=[]
    for label,x in [('absolute_D',abs(grid.D)),('late_minus_early',grid.late_minus_early)]:
        correlations.append(dict(x='level_gap_pp',y=label,n=6,rho=stats.spearmanr(grid.level_gap_pp,x).statistic,note='n=6, descriptive only; no generalization to model populations'))
    table(pd.DataFrame(correlations),'03_rank_correlations')
    fig,ax=plt.subplots(figsize=(7,4))
    for _,r in grid.iterrows():
        ax.scatter(r.level_gap_pp,r.D,marker='v' if r.persistence_class=='decays' else 'o',s=65)
        ax.annotate(r.model,(r.level_gap_pp,r.D),xytext=(5,5),textcoords='offset points',fontsize=8)
    ax.axhline(0,color='gray',ls=':'); ax.set_xlabel('Level gap F − S (pp)'); ax.set_ylabel('Four-point interaction D (pp)'); ax.set_title('POST-HOC / EXPLORATORY\nCircle: persists; triangle: decays (sign of late − early)'); fig.tight_layout(); fig.savefig(HERE/'figures/03_level_slope_persistence.pdf'); plt.close(fig)
    text='All uncertainty resamples episodes within model-mode-N cells (5,000 draws), averaging four N cells equally. Slopes are episode OLS slopes on rounds 1–30, then averaged. Zero-late means every late-round seat action is 4. Crossing is strictly below half the opening mean; zero opening yields a zero threshold and never. Never bootstrap outcomes are reported separately from finite-crossing quantiles.\n\n'
    text+=grid[['model','level_gap_pp','D','late_minus_early','late_minus_early_low','late_minus_early_high','fraction_zero_late','half_opening_round']].to_string(index=False)+'\n\n'
    text+=pd.DataFrame(correlations).to_string(index=False)+'\n\n'
    text+='Verdict: level, N scaling and temporal persistence describe different behavior. Mistral has high sustained cooperation; Qwen and GPT-OSS have zero late cooperation in every episode, and DeepSeek shows a large mean decline. Llama has a small descriptive decline, while Gemma rises on average. The figure classes use only the sign of late-minus-early and do not imply a significant change; consult bootstrap CIs. Rank correlations involve only six named models and are descriptive only.'
    section('Task 3 — persistence and summary grid',text); print(grid[['model','level_gap_pp','D','late_minus_early','late_minus_early_low','late_minus_early_high','fraction_zero_late']].to_string(index=False))

if __name__=='__main__': main()
