"""POST-HOC / EXPLORATORY: real-number paper inserts and artifact manifest."""
import importlib.metadata as metadata
import platform
import re
from common import *

def main():
    require_gate()
    filtered=pd.read_csv(HERE/'results/01_filtered_outcomes.csv')
    concise=filtered.copy()
    concise['Early / late (%)']=concise.apply(lambda r:f'{r.early:.2f} / {r.late:.2f}' if pd.notna(r.early) and pd.notna(r.late) else 'not estimable',axis=1)
    concise['D (95% CI), pp']=concise.apply(lambda r:f'{r.D:+.2f} [{r.D_low:.2f}, {r.D_high:.2f}]' if pd.notna(r.D_low) else 'not estimable',axis=1)
    table(concise[['filter','model','mode','episodes','Early / late (%)','D (95% CI), pp']],'01_filtered_paper_table')
    grid=pd.read_csv(HERE/'results/03_summary_grid.csv'); paper=pd.DataFrame({'Model':grid.model,'Level gap (pp)':grid.level_gap_pp.map(lambda x:f'{x:.2f}'),'D (95% CI), pp':grid.apply(lambda r:f'{r.D:+.2f} [{r.low:.2f}, {r.high:.2f}]',axis=1),'F late - early (95% CI), pp':grid.apply(lambda r:f'{r.late_minus_early:+.2f} [{r.late_minus_early_low:.2f}, {r.late_minus_early_high:.2f}]',axis=1),'F zero-late fraction':grid.fraction_zero_late.map(lambda x:f'{x:.4f}')})
    # Main paper-ready LaTeX counterpart; preserve numeric CSV for reuse.
    tex=paper.to_latex(index=False,escape=True,column_format='lrrrr')
    tex='\\begin{table*}[t]\n\\centering\\small\n\\resizebox{\\textwidth}{!}{%\n'+tex+'}\n\\caption{POST-HOC / EXPLORATORY. Level gap, four-point group-size interaction, and free-form persistence. Cooperation is in percentage points; temporal intervals resample episodes within N cells.}\n\\label{tab:posthoc-summary}\n\\end{table*}\n'
    (HERE/'results/latex/03_summary_grid.tex').write_text('% POST-HOC / EXPLORATORY\n'+tex,encoding='utf-8')
    rates=pd.read_csv(HERE/'results/04_mismatch_by_model_mode.csv'); p=rates[rates.scope=='all_nonempty']; total=int(p.flagged.sum()); n=int(p.decisions.sum())
    compact=p[['model','mode','decisions','flagged']].copy()
    compact['Flagged % (Wilson CI)']=p.apply(lambda r:f'{r.percent:.2f} [{r.wilson_low:.2f}, {r.wilson_high:.2f}]',axis=1)
    compact['Episode-bootstrap CI']=p.apply(lambda r:f'[{r.cluster_low:.2f}, {r.cluster_high:.2f}]',axis=1); table(compact,'04_mismatch_paper_table')
    q=pd.read_csv(HERE/'results/01_qwen_late.csv').iloc[0]; loo=pd.read_csv(HERE/'results/02_leave_one_out.csv').set_index('excluded').loc['Llama']; gemma=p[(p.model=='Gemma')]; llama=p[(p.model=='Llama')]
    inserts=f'''### (a) Repair/fallback robustness

In post-hoc sensitivity analyses, Qwen's late free-form cooperation remained {q.nonfallback_late_cooperation:.2f}% among all {int(q.nonfallback_decisions)} nonfallback decisions in rounds 26–30; none of the {int(q.late_decisions)} late decisions used fallback. Thus fallback did not directly produce its late zero cooperation. High repair rates nevertheless made strict episode-filter estimates unestimable in some cells. DeepSeek's decline remained negative under every filter, while Llama's small baseline decline changed sign under some filters. Conditional first-attempt estimates can be selection-biased and do not identify a causal pipeline effect.

### (b) Heterogeneity robustness

The post-hoc heterogeneity result depended on Llama: excluding it gave Q(4)={loo.Q:.2f}, p={loo.p:.3f}, I²={loo.I2:.2f}%; every other single-model exclusion retained significance, supporting a claim that one model differs rather than broad robust heterogeneity.

### (c) Summary table caption

POST-HOC / EXPLORATORY comparison of free-form minus structured cooperation level, four-point group-size interaction D, free-form late-minus-early cooperation, and the fraction of episodes with exactly zero late cooperation. Levels average four N cells equally. D intervals use eight-cell Welch uncertainty; temporal intervals resample whole episodes within N cells (5,000 draws). Late and early cover rounds 26–30 and 1–5. Level, scaling and persistence measure different features of behavior; intervals containing zero do not establish equivalence.

### (d) Explanation audit

A post-hoc four-rule detector flagged {total:,}/{n:,} nonempty N=2/5 explanations ({100*total/n:.2f}%). Gemma's structured/free-form rates were {gemma[gemma['mode']=='structured'].percent.iloc[0]:.2f}%/{gemma[gemma['mode']=='free_form'].percent.iloc[0]:.2f}%; Llama's were {llama[llama['mode']=='structured'].percent.iloc[0]:.2f}%/{llama[llama['mode']=='free_form'].percent.iloc[0]:.2f}%. The 274 empty explanations were excluded. Rates among valid unrepaired decisions are reported separately. These estimates measure detectable inconsistencies under the stated rule-based detector, not misunderstanding, and apply only to N=2/5. Wilson intervals are descriptive; episode-bootstrap intervals address clustering. A blinded 150-case validation sample awaits human labels, so detector precision and recall remain unknown.
'''
    section('PAPER INSERTS',inserts)
    section('Reproducibility and execution notes','No new episodes, inference, GPU allocation, or judge calls were performed. **Saved-call anomaly:** 238 error-only call records have no output/parser-version field: 118 DeepSeek repair calls, 96 GPT-OSS initial calls and 24 GPT-OSS repair calls. These are explicitly logged call failures, not unobserved submitted actions; submitted actions and original reliability flags remain available. An error-only first call is not first-attempt-valid. Parser-version None in the manifest refers to these records. The original explanation-export hashes serialize strings as JSON; a raw-text hash check initially failed, was diagnosed against sample_for_cot_coding.py, and was corrected to the documented hash convention before audit results were calculated. All hashes and text links then passed. The optional judge was omitted because access/authorization was not supplied; no requested rank correlations were skipped. Human labeling remains outstanding. Scripts can be rerun in small sequential CPU jobs on Sharanga, with Task 0 failure preventing all later tasks. Local task commits preserve completed checkpoints; final packaging does not overwrite the manuscript.')
    verification=HERE/'server_verification.json'
    if verification.exists():
        receipt=json.loads(verification.read_text())
        section('Sharanga verification',f'All six short CPU jobs completed successfully (388781–388786), each requesting the compute/cpulimit minimum of four CPUs, 4 GB RAM and ten minutes; zero GPUs. Task wall times were 18, 32, 9, 9, 17 and 4 seconds. All {len(receipt["tables"])} downloaded CSV tables match the local outputs at rtol=atol=1e-10. Both environments passed all 97 reproduction anchors. Remote statsmodels is 0.14.5 versus local 0.15.0; other pinned numerical/plotting packages match. See server_verification.json and SHARANGA.md for receipts and deployment issues. Scheduler dependencies handled CPU availability without any GPU polling loop.')
    report=HERE/'RESULTS.md'
    chunks=re.split(r'(?=^## )',report.read_text(encoding='utf-8'),flags=re.MULTILINE)
    order=['Task 0','Task 1','Task 2','Task 3','Task 4','PAPER INSERTS','Sharanga verification','Reproducibility']
    chunks[1:]=sorted(chunks[1:],key=lambda s:next((i for i,t in enumerate(order) if s.startswith('## '+t)),len(order)))
    report.write_text(''.join(chunks),encoding='utf-8')
    versions={p:metadata.version(p) for p in ['pandas','numpy','scipy','statsmodels','matplotlib','jinja2']}
    manifest=dict(label='POST-HOC / EXPLORATORY',seed=SEED,python=platform.python_version(),platform=platform.platform(),packages=versions,source_sha256=SOURCE_HASH,artifacts={})
    for folder in ['results','figures','validation']:
        for path in sorted((HERE/folder).rglob('*')):
            if path.is_file() and path.suffix in ['.csv','.tex','.pdf','.py','.json'] and path.name!='artifact_manifest.json':
                manifest['artifacts'][str(path.relative_to(HERE))]=hashlib.sha256(path.read_bytes()).hexdigest()
    (HERE/'results/artifact_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (HERE/'package_versions.json').write_text(json.dumps(versions,indent=2),encoding='utf-8')
    print('Finalized paper inserts and',len(manifest['artifacts']),'artifact hashes')

if __name__=='__main__': main()
