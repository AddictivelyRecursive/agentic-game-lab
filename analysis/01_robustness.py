"""POST-HOC / EXPLORATORY: episode-aware repair/fallback sensitivity."""
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import *

def main():
    require_gate(); ap=argparse.ArgumentParser(); ap.add_argument('--input',default=DEFAULT_INPUT); a=ap.parse_args()
    ep,dd,rr=load(a.input)
    rate=dd.groupby(['model','mode','round']).agg(decisions=('c','size'),fallback_rate=('fallback','mean'),repair_rate=('repair','mean')).reset_index()
    coop=rr.groupby(['model','mode','N','round']).c.mean().groupby(['model','mode','round']).mean().reset_index(name='cooperation')
    rate=rate.merge(coop); table(rate,'01_fallback_by_round')
    for (m,mode),s in rate.groupby(['model','mode']):
        fig,ax=plt.subplots(1,2,figsize=(9,3)); ax[0].plot(s['round'],100*s.fallback_rate,label='Fallback'); ax[0].plot(s['round'],100*s.repair_rate,label='Repair'); ax[0].legend()
        ax[1].plot(s['round'],100*s.cooperation); ax[0].set_ylabel('Seat-decision rate (%)'); ax[1].set_ylabel('Cooperation (%)')
        for x in ax: x.set_xlabel('Round'); x.set_ylim(0,100)
        fig.suptitle(f'POST-HOC / EXPLORATORY — {m} {mode}'); fig.tight_layout(); fig.savefig(HERE/'figures'/f'01_{m}_{mode}.pdf'); plt.close(fig)
    filters={'unfiltered':np.ones(len(ep),bool),'no_fallback':ep.fallback.eq(0),'repair_le_25':ep.repair.le(.25),'repair_le_10':ep.repair.le(.10)}
    outcomes=[]; counts=[]; effects=[]
    for name,mask in filters.items():
        sub=ep[mask]; eff=contrast(sub); eff['filter']=name; effects.append(eff)
        cnt=sub.groupby(['model','mode','N']).size()
        for (m,mode,n),base in ep.groupby(['model','mode','N']):
            kept=int(cnt.get((m,mode,n),0)); counts.append(dict(filter=name,model=m,mode=mode,N=n,retained=kept,total=len(base),below_two=kept<2))
        for (m,mode),base in ep.groupby(['model','mode']):
            s=sub[(sub.model==m)&(sub['mode']==mode)]; cells=s.groupby('N')[['c','early','late']].mean()
            valid=len(cells)==4; vals=cells.mean()*100 if valid else pd.Series({c:np.nan for c in ['c','early','late']})
            d=eff[eff.model==m].iloc[0]
            outcomes.append(dict(filter=name,model=m,mode=mode,episodes=len(s),cells=len(cells),cooperation=vals.c,early=vals.early,late=vals.late,late_minus_early=vals.late-vals.early,D=d.D,D_low=d.low,D_high=d.high,D_status=d.status))
    # Conditional decision estimates: no eligible decisions => explicit missing episode/cell.
    eligible=dd[dd.first_valid&~dd.repair&~dd.fallback]
    conditional=[]
    for (m,mode),base in dd.groupby(['model','mode']):
        selected=eligible[(eligible.model==m)&(eligible['mode']==mode)]; row=dict(filter='first_attempt_valid',model=m,mode=mode,episodes=selected.episode_id.nunique(),eligible_decisions=len(selected),total_decisions=len(base))
        for label,lo,hi in [('cooperation',1,30),('early',1,5),('late',26,30)]:
            s=selected[selected['round'].between(lo,hi)].groupby(['N','episode_id']).c.mean().groupby('N').mean()
            row[label]=100*s.mean() if len(s)==4 else np.nan
            row[label+'_eligible_episodes']=selected[selected['round'].between(lo,hi)].episode_id.nunique()
            row[label+'_eligible_cells']=len(s)
        row['late_minus_early']=row['late']-row['early']; row['D_status']='conditional estimate; not episode-filter D'
        conditional.append(row)
    allout=pd.DataFrame(outcomes+conditional); table(allout,'01_filtered_outcomes'); table(pd.DataFrame(counts),'01_retained_per_cell'); table(pd.concat(effects),'01_filtered_D')
    dose=[]
    for (m,mode),s in ep.groupby(['model','mode']):
        groups=[g for _,g in s.groupby('N')]; random=rng(f'dose {m} {mode}')
        indexes=np.concatenate([random.integers(0,len(g),(5000,len(g)))+sum(len(h) for h in groups[:i]) for i,g in enumerate(groups)],axis=1)
        ordered=pd.concat(groups); y=ordered.late.to_numpy()
        for predictor in ['fallback','repair']:
            x=ordered[predictor].to_numpy(); valid=np.ptp(x)>0 and np.ptp(y)>0
            rho=stats.spearmanr(x,y).statistic if valid else np.nan; boot=[]
            if valid:
                for ix in indexes:
                    if np.ptp(x[ix])>0 and np.ptp(y[ix])>0: boot.append(stats.spearmanr(x[ix],y[ix]).statistic)
            low,high=np.quantile(boot,[.025,.975]) if boot else (np.nan,np.nan)
            dose.append(dict(model=m,mode=mode,predictor=predictor,rho=rho,low=low,high=high,episodes=len(s),draws=5000,estimable_draws=len(boot),invalid_draws=5000-len(boot),status='estimated; CI conditional on estimable draws' if boot else 'not estimable: constant predictor or outcome'))
    table(pd.DataFrame(dose),'01_dose_response')
    late=dd[(dd.model=='Qwen')&(dd['mode']=='free_form')&dd['round'].between(26,30)]
    q=pd.DataFrame([dict(model='Qwen',mode='free_form',late_decisions=len(late),fallbacks=int(late.fallback.sum()),fallback_percent=100*late.fallback.mean(),nonfallback_decisions=int((~late.fallback).sum()),nonfallback_late_cooperation=100*late[~late.fallback].c.mean())]); table(q,'01_qwen_late')
    verdicts=[]
    for m in MODELS.values():
        s=allout[(allout.model==m)&(allout['mode']=='free_form')]; base=s[s['filter']=='unfiltered'].iloc[0]; checks=s[s['filter']!='unfiltered']; finite=checks.dropna(subset=['early','late'])
        if base.late_minus_early>=0: label='DOES NOT SURVIVE'; reason='No baseline early-then-decay pattern exists.'
        elif len(finite)<len(checks): label='NOT ESTIMABLE'; reason='At least one filtered mode summary lacks an N cell; observed conditional estimates cannot resolve robustness across all filters.'
        elif (finite.late_minus_early<0).all(): label='SURVIVES'; reason='The late-minus-early sign remains negative under every estimable filter; this is descriptive sign robustness, not equivalence of magnitudes.'
        elif (finite.late_minus_early>=0).all(): label='DOES NOT SURVIVE'; reason='All filtered summaries remove or reverse the baseline decline.'
        else: label='WEAKENED'; reason='Filtering produces inconsistent decline signs.'
        verdicts.append(dict(model=m,label=label,reason=reason,baseline_early=base.early,baseline_late=base.late))
    table(pd.DataFrame(verdicts),'01_verdicts')
    text='Original repair/fallback flags define episode filters. Four N cells receive equal weight; retained-cell counts and <2 flags are in 01_retained_per_cell.csv. D is not estimable when any of eight cells has <2 episodes. First-attempt estimates are conditional and may be selection-biased. Spearman CIs resample episodes within N; undefined draws are counted and reported.\n\n'
    text+=f'Qwen F late fallbacks: {int(late.fallback.sum())}/{len(late)} = {100*late.fallback.mean():.2f}%; nonfallback late cooperation = {100*late[~late.fallback].c.mean():.2f}% ({int((~late.fallback).sum())} decisions). Thus late zero cooperation is present among nonfallback decisions too.\n\n'
    for v in verdicts:
        text+=f'**{v["model"]}: {v["label"]}.** Baseline early/late {v["baseline_early"]:.2f}/{v["baseline_late"]:.2f}%. {v["reason"]}\n'
        s=allout[(allout.model==v['model'])&(allout['mode']=='free_form')]
        text+='; '.join(f'{r["filter"]}: {r["early"]:.2f}/{r["late"]:.2f}% (n={r["episodes"]})' for r in s.to_dict('records'))+'.\n\n'
    text+='Interpretation: no-fallback and low-repair restrictions select episodes and can change their composition. A sign-only SURVIVES verdict does not establish a mechanism; NOT ESTIMABLE is not evidence against decay. Constant late outcomes make several dose-response correlations undefined. For Gemma and Mistral, DOES NOT SURVIVE denotes absence of baseline decay, not a filtering-induced reversal.'
    section('Task 1 — repair/fallback robustness',text)
    print(q.to_string(index=False)); print(pd.DataFrame(verdicts).to_string(index=False))

if __name__=='__main__': main()
