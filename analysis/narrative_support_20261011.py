"""Exploratory saved-log descriptions; no inference, model calls, or significance tests."""
import hashlib
import json
import re
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'results/server_copies/final_main_20261010/main_complete_20261010/main.jsonl'
OUT = ROOT/'results/analysis/narrative_support_20261011'
EXPECTED = '149ac53f8a19286b429d1de83939ae91028845b0b35a18e1dc8dd5cc0468b778'
MODELS = dict(zip(['llama-3.1-8b-instruct','qwen3-8b','gemma-3-12b-it',
    'mistral-small-3.1-24b-instruct','deepseek-r1-distill-qwen-14b','gpt-oss-120b'],
    ['Llama','Qwen','Gemma','Mistral','DeepSeek','GPT-OSS']))
SEEDS = dict(zip(MODELS.values(), [14,4,8,3,16,2]))
PATTERNS = {
    'ranking': r'\brank(?:ing|ed|s)?\b|\bexpected utility\b',
    'horizon': r'\b(?:rounds? remaining|remaining rounds?|last rounds?|final rounds?|end[- ]game|horizon|total rounds?|rounds? left)\b|\b(?:30|thirty)[- ]round\b|\bround\s+30\b',
}

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def longpath(path):
    return Path('\\\\?\\'+str(path.resolve())) if sys.platform=='win32' else path

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    assert sha(SOURCE)==EXPECTED, 'Source differs from original complete-main dataset'
    episodes=[]; rounds=[]; decisions=[]; episode_ids=set()
    with SOURCE.open(encoding='utf8') as f:
        for line in f:
            row=json.loads(line)
            if 'decisions' not in row: continue
            eid=row['episode_id']; assert eid not in episode_ids; episode_ids.add(eid)
            model=MODELS[row['model']]; mode=row['reasoning_mode']; n=row['N']
            assert row['status']=='completed' and n in range(2,6)
            ident=dict(episode_id=eid,model=model,mode=mode,N=n,seed=row['seed'])
            rs=sorted(row['rounds'],key=lambda r:r['t'])
            assert [r['t'] for r in rs]==list(range(1,31))
            c=np.array([np.mean(r['true_coop']) for r in rs])
            assert np.isclose(c.mean(),row['cooperation'])
            breaks=[]; streak=0
            for r,g in zip(rs,c):
                assert r['streak_prev']==streak
                assert r['true_coop']==[1-a/4 for a in r['true_actions']]
                active_break=streak>0 and g<.5
                if active_break: breaks.append(r['t'])
                streak=streak+1 if g>=.5 else 0
                assert r['streak_next']==streak
                rounds.append(dict(**ident,round=r['t'],cooperation=g,
                    streak_prev=r['streak_prev'],streak_next=streak,active_break=active_break))
            below=np.flatnonzero(c<.5)
            # Terminal zero run is an observed action pattern, not a fitted change point.
            zero_start=next((t for t in range(1,31) if np.all(c[t-1:]==0)),None)
            first_break=breaks[0] if breaks else None
            post=c[first_break:min(first_break+3,30)] if first_break else np.array([])
            before=c[max(0,first_break-4):first_break-1] if first_break else np.array([])
            episodes.append(dict(**ident,early=c[:5].mean(),late=c[-5:].mean(),
                first_below=int(below[0]+1) if len(below) else None,
                first_active_break=first_break,break_count=len(breaks),
                any_cooperation=bool(np.any(c>0)),terminal_zero_start=zero_start,
                terminal_zero_from_round1=zero_start==1,
                terminal_zero_after_opening=zero_start is not None and zero_start>1,
                terminal_zero_late_onset=zero_start is not None and zero_start>=26,
                first_break_pre_mean=float(before.mean()) if len(before) else None,
                first_break_next3_mean=float(post.mean()) if len(post) else None,
                recovered_threshold_after_first_break=bool(np.any(post>=.5)) if len(post) else None))
            seen=set()
            for d in row['decisions']:
                assert (d['round'],d['seat']) not in seen; seen.add((d['round'],d['seat']))
                text=d['reasoning_text']; assert isinstance(text,str)
                assert not d['fallback_used'] or rs[d['round']-1]['true_actions'][d['seat']]==4
                flags={k:bool(re.search(v,text,flags=re.I)) for k,v in PATTERNS.items()}
                decisions.append(dict(**ident,round=d['round'],seat=d['seat'],
                    nonempty=bool(text.strip()),fallback=d['fallback_used'],**flags))
            assert len(seen)==30*n
    ep=pd.DataFrame(episodes); rr=pd.DataFrame(rounds); dd=pd.DataFrame(decisions)
    assert len(ep)==376 and len(dd)==39480
    cells=ep.groupby(['model','mode','N']).size()
    assert len(cells)==48
    assert all(k==SEEDS[m] for (m,mode,n),k in cells.items())
    ep.to_csv(OUT/'episode_events.csv',index=False)
    dd.to_csv(OUT/'keyword_flags.csv',index=False)

    # Verify the supplied horizon from every saved per-seat trace, not just wrapper code.
    final=ROOT/'results/server_copies/final_main_20261010'
    snapshot=ROOT/'results/server_copies/snapshot_20261009_2103/results/sharanga/main_short_chunks_20261007/shards'
    checked=0; checked_ids=set(); trace_sources=[]
    for source in json.loads((final/'main_complete_20261010/source_checksums.json').read_text()):
        worker=source['worker']; model=worker.split('__chunk')[0]
        base=final/'remaining_shards' if model.startswith('deepseek') and worker.endswith(('chunk14','chunk15')) else snapshot
        path=longpath(base/worker/model/'output/main.jsonl')
        assert sha(path)==source['sha256']
        trace_sources.append(dict(worker=worker,sha256=source['sha256']))
        with path.open(encoding='utf8') as f:
            for line in f:
                row=json.loads(line)
                if 'decisions' not in row: continue
                eid=row['episode_id']; assert eid in episode_ids and eid not in checked_ids; checked_ids.add(eid)
                for seat in range(row['N']):
                    p=path.parent/'episodes'/eid/'agents'/f'p{seat}'/'agent_traces.jsonl'
                    seen=set()
                    with p.open(encoding='utf8') as tf:
                        for tline in tf:
                            state=json.loads(tline); t=state['round']; assert t not in seen; seen.add(t)
                            assert state['raw_input']['temporal_context']==dict(total_rounds=30,rounds_remaining_including_current=31-t)
                            checked+=1
                    assert seen==set(range(1,31))
    assert checked_ids==episode_ids and checked==39480

    summaries=[]; curves=[]; keywords=[]
    for (m,mode),e in ep.groupby(['model','mode'],sort=False):
        sub=rr[(rr.model==m)&(rr['mode']==mode)]
        by_n=sub.groupby(['N','round']).cooperation.mean().unstack('round')
        mean=by_n.mean().to_numpy()
        # Independent episode trajectories within each N; four cells equally weighted.
        se=np.sqrt(sum(sub[sub.N==n].pivot(index='episode_id',columns='round',values='cooperation').var(ddof=1).to_numpy()/SEEDS[m] for n in range(2,6))/16)
        first=np.flatnonzero(mean<.5)
        event=e[e.first_active_break.notna()]
        eligible=event[event.first_break_next3_mean.notna()]
        collapsing=e[e.terminal_zero_after_opening]
        summaries.append(dict(model=m,mode=mode,episodes=len(e),
            curve_first_below=int(first[0]+1) if len(first) else None,
            episodes_below_round1=int((e.first_below==1).sum()),
            episodes_with_active_break=len(event),
            first_active_break_median=event.first_active_break.median(),
            first_active_break_q1=event.first_active_break.quantile(.25),
            first_active_break_q3=event.first_active_break.quantile(.75),
            first_break_with_later_observation=len(eligible),
            first_break_pre_mean=eligible.first_break_pre_mean.mean(),
            first_break_next3_mean=eligible.first_break_next3_mean.mean(),
            episodes_recovered_next3=int(eligible.recovered_threshold_after_first_break.sum()),
            terminal_zero_from_round1=int(e.terminal_zero_from_round1.sum()),
            terminal_zero_after_opening=len(collapsing),
            terminal_zero_start_median=collapsing.terminal_zero_start.median(),
            terminal_zero_late_onset=int(e.terminal_zero_late_onset.sum()),
            terminal_zero_not_reached=int(e.terminal_zero_start.isna().sum())))
        for t,(g,s) in enumerate(zip(mean,se),1): curves.append(dict(model=m,mode=mode,round=t,mean=g,se=s))
        d=dd[(dd.model==m)&(dd['mode']==mode)]
        kw=dict(model=m,mode=mode,decisions=len(d),nonempty=int(d.nonempty.sum()),fallbacks=int(d.fallback.sum()))
        for label in PATTERNS:
            kw[label+'_count']=int(d[label].sum())
            kw[label+'_all_percent']=100*d[label].mean()
            kw[label+'_nonempty_percent']=100*d.loc[d.nonempty,label].mean()
            # An episode/stratum-balanced alternative to seat-weighted text counts.
            kw[label+'_episode_balanced_percent']=100*d.groupby(['N','episode_id'])[label].mean().groupby('N').mean().mean()
        keywords.append(kw)
    summary=pd.DataFrame(summaries); curve=pd.DataFrame(curves); keyword=pd.DataFrame(keywords)
    summary.to_csv(OUT/'event_summary.csv',index=False); curve.to_csv(OUT/'round_curves.csv',index=False); keyword.to_csv(OUT/'keyword_summary.csv',index=False)
    def fmt(x): return '--' if pd.isna(x) else f'{x:g}'
    timing_rows=[]; keyword_rows=[]
    for r in summary.to_dict('records'):
        timing_rows.append([r['model'],'S' if r['mode']=='structured' else 'F',r['episodes'],
            r['episodes_with_active_break'],r['first_active_break_median'],r['episodes_recovered_next3'],
            r['terminal_zero_from_round1'],r['terminal_zero_after_opening'],r['terminal_zero_start_median'],r['terminal_zero_late_onset']])
    for r in keyword.to_dict('records'):
        keyword_rows.append([r['model'],'S' if r['mode']=='structured' else 'F',r['decisions'],r['nonempty'],
            r['ranking_count'],f"{r['ranking_all_percent']:.2f}",r['horizon_count'],f"{r['horizon_all_percent']:.2f}"])
    def tex_table(filename,columns,headers,rows,caption,label):
        lines=['\\begin{table*}[t]','\\centering\\small',r'\setlength{\tabcolsep}{3pt}',
            '\\begin{tabular}{'+columns+'}',r'\toprule',' & '.join(headers)+r' \\',r'\midrule']
        for row in rows: lines.append(' & '.join(fmt(x) if isinstance(x,(float,int)) else str(x) for x in row)+r' \\')
        lines.extend([r'\bottomrule',r'\end{tabular}','\\caption{'+caption+'}','\\label{'+label+'}',r'\end{table*}'])
        (OUT/filename).write_text('\n'.join(lines)+'\n',encoding='utf8')
    tex_table('log_timing.tex','llrrrrrrrr',
        ['Model','Mode','$n$','Breaks',r'\shortstack{Break\\median $t$}','Recovery','Floors','Terminal',r'\shortstack{Onset\\median $t$}','Late'],
        timing_rows,
        r'Post-hoc episode timing in the original cohort. S/F denote structured/free-form. Breaks count episodes with an active streak reset; Recovery counts return to $g_t\geq.5$ within the next three rounds after the first break. Floors are zero throughout all 30 rounds. Terminal counts episodes with an all-zero suffix beginning after round 1; Late counts these onsets in rounds 26--30. Medians condition on the respective event; dashes denote no events. Counts are descriptive and do not identify mechanisms.',
        'tab:log-timing')
    tex_table('log_keywords.tex','llrrrrrr',
        ['Model','Mode','Decisions','Nonempty',r'Ranking $n$',r'Ranking (\%)',r'Horizon $n$',r'Horizon (\%)'],keyword_rows,
        r'Post-hoc literal keyword matches in returned explanations. Percentages use all seat-round decisions, including empty explanations as nonmatches; Nonempty reports the number with nonblank text. S/F denote structured/free-form. Categories may overlap. These decision-weighted lexical counts are not validated semantic labels, estimates of hidden reasoning, or independent behavioral replicates.',
        'tab:log-keywords')
    fig,axes=plt.subplots(2,3,figsize=(10,5.4),sharex=True,sharey=True)
    for ax,m in zip(axes.flat,MODELS.values()):
        for mode,color,label in [('structured','#00728c','S'),('free_form','#bd4b20','F')]:
            data=curve[(curve.model==m)&(curve['mode']==mode)]
            x=data['round'].to_numpy(); y=data['mean'].to_numpy(); se=data['se'].to_numpy()
            ax.plot(x,y,color=color,label={'S':'structured','F':'free-form'}[label],lw=1.4)
            ax.fill_between(x,np.clip(y-se,0,1),np.clip(y+se,0,1),color=color,alpha=.14)
            hit=np.flatnonzero(y<.5)
            if len(hit): ax.plot(x[hit[0]],y[hit[0]],marker='x',ms=7,mew=1.6,color=color,clip_on=False)
        ax.axhline(.5,color='.5',ls='--',lw=.7)
        ax.set(title=m,ylim=(-.035,1.02),xlim=(.5,30.5),xticks=[1,10,20,30])
        ax.grid(alpha=.12)
    for ax in axes[:,0]: ax.set_ylabel('Mean cooperation')
    for ax in axes[1]: ax.set_xlabel('Round')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False)
    fig.tight_layout(rect=(0,.055,1,1)); fig.savefig(OUT/'main_dynamics_threshold.pdf'); plt.close(fig)
    manifest=dict(status='POST-HOC / EXPLORATORY',source_sha256=EXPECTED,episodes=len(ep),decisions=len(dd),
        horizon_observations_checked=checked,trace_sources=trace_sources,keyword_patterns=PATTERNS,
        script_sha256=sha(Path(__file__)),versions=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__,matplotlib=matplotlib.__version__),
        definitions=dict(active_break='g_t < .5 and streak_prev > 0; round 1 below threshold is not an active break',
        curve_crossing='first group-size-balanced model-mode round mean strictly below .5; not an episode break',
        terminal_zero='first round of an all-zero suffix through round 30; report round-1 floors separately',
        event_window='up to 3 available rounds before first active break, excluding break; next up to 3 rounds excluding break; eligible episodes only, descriptive selection',
        keywords='case-insensitive literal regex on saved reasoning_text, including repair-returned explanations; no inferred reasoning, judge, semantic annotation, or significance tests'))
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    print(summary.to_string(index=False)); print(keyword.to_string(index=False)); print('Verified horizon observations:',checked)

if __name__=='__main__': main()
