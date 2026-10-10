"""POST-HOC / EXPLORATORY: reproduction gate, no inference calls."""
import argparse
import json
import numpy as np
import pandas as pd
from common import *

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',default=DEFAULT_INPUT); args=ap.parse_args()
    ep,dd,rr=load(args.input)
    (HERE/'SCHEMA.md').write_text('''# POST-HOC / EXPLORATORY — verified schema

Canonical input and SHA256: see results/source_manifest.json. JSONL includes manifest and episode records; only completed episode records enter analysis. All 376 episodes / 48 cells / 39,480 decisions verified; seed counts 14/4/8/3/16/2 per model-mode-N cell. Unique episode_id and (model,mode,N,seed); exactly 30*N distinct seat-round decisions per episode. No pilots included.

Episode fields: model (mapped explicitly to short name), reasoning_mode (structured/free_form), N, seed, episode_id, status. Decision fields: round (1–30), seat (0–N−1), repair_used (bool), repair_calls (0–3), fallback_used (bool), reasoning_text, first_parse_ok (historical proxy), model_calls, aborted. Submitted action is absent directly on decision: join by round to rounds[t].true_actions[seat]. Verified integer 0–4 and c=1−a/4 against true_coop and stored episode cooperation. Every fallback action is 4. No aborted decisions.

Use original repair_used/fallback_used flags exactly as Table 5; never proxy these from action or call count. Recompute first_parse and first_valid from first model_calls output using the saved v2 extractor and validate_decision, allowing digit strings, excluding booleans, requiring integer 0–4. Extraction success alone does not imply action validity. Retain historical first_parse_ok separately. Parser versions and call stages are listed in the manifest. Explanation is reasoning_text; empty means whitespace-only. Missing required fields, duplicate joins or source hash mismatch raise an error before results.

Unit of inference: episode. Model-mode outcome means average four N-cell episode means equally. Seat-decision repair/fallback rates are descriptive. Cooperation is stored internally in fractions, reported as percent/pp. No new model inference.
''',encoding='utf-8')
    table(ep,'00_episode_outcomes'); table(ep.groupby(['model','mode','N']).agg(episodes=('c','size'),mean=('c','mean'),variance=('c','var')).reset_index(),'00_cells')
    effects=contrast(ep); table(effects,'00_effects'); h=heterogeneity(effects); table(pd.DataFrame([h]),'00_heterogeneity')
    rows=[]
    def check(q,pub,value,tol): rows.append(dict(quantity=q,published=pub,recomputed=value,abs_diff=abs(pub-value),**{'pass':abs(pub-value)<=tol+1e-10}))
    levels={'Llama':(6.52,20.43),'Qwen':(.14,6.24),'Gemma':(0,13.02),'Mistral':(6.75,80.24),'DeepSeek':(1.19,18.24),'GPT-OSS':(.26,2.72)}
    temporal={'Llama':(23.41,21.94),'Qwen':(22.57,0),'Gemma':(13.53,17.02),'Mistral':(78.08,80.30),'DeepSeek':(39.87,2.79),'GPT-OSS':(12.54,0)}
    rates={'Llama':((4.5,.4),(10.4,1.3)),'Qwen':((68.5,0),(92.1,6)),'Gemma':((0,0),(0,0)),'Mistral':((1.2,0),(.6,0)),'DeepSeek':((.7,0),(2.1,.1)),'GPT-OSS':((6.5,0),(16.7,.7))}
    ds={'Llama':(14.02,4.67,23.37,.004),'Qwen':(-3.91,-8.54,.73,.084),'Gemma':(-5.79,-16.11,4.52,.247),'Mistral':(-6.81,-16.93,3.31,.151),'DeepSeek':(5.73,-3.32,14.78,.208),'GPT-OSS':(-7.87,-116.99,101.24,.542)}
    summary=means(ep,('c','early','late','opening'))
    for model in MODELS.values():
        for i,mode in enumerate(MODES):
            r=summary[(summary.model==model)&(summary['mode']==mode)].iloc[0]
            check(f'{model} {mode} level',levels[model][i],100*r.c,.01)
            d=dd[(dd.model==model)&(dd['mode']==mode)]
            check(f'{model} {mode} decisions',COUNTS[model]*420,len(d),0)
            for j,col in enumerate(['repair','fallback']): check(f'{model} {mode} {col} %',rates[model][i][j],100*d[col].mean(),.05)
            if mode=='free_form':
                for j,col in enumerate(['early','late']): check(f'{model} F {col}',temporal[model][j],100*r[col],.01)
        e=effects[effects.model==model].iloc[0]
        for col,pub,tol in zip(['D','low','high','p'],ds[model],[.05,.005,.005,.0005]): check(f'{model} {col}',pub,e[col],tol)
        s=ep[ep.model==model]
        slopes=[np.polyfit(s[s['mode']==mode].N,s[s['mode']==mode].c*100,1)[0] for mode in MODES]
        assert np.isclose(3*(slopes[1]-slopes[0]),e.D)
    check('Llama Holm p',.024,effects.set_index('model').loc['Llama','holm'],.0005)
    check('Qwen F opening',59.11,100*summary[(summary.model=='Qwen')&(summary['mode']=='free_form')].opening.iloc[0],.01)
    for n,pub in zip(range(2,6),[14.53,6.63,30.91,0]): check(f'Gemma F N={n}',pub,100*ep[(ep.model=='Gemma')&(ep['mode']=='free_form')&(ep.N==n)].c.mean(),.01)
    for col,pub,tol in [('Q',18.23,.005),('p',.0027,.00005),('I2',72.6,.05)]: check('heterogeneity '+col,pub,h[col],tol)
    check('total episodes',376,len(ep),0); check('total decisions',39480,len(dd),0)
    export=dd[dd.N.isin([2,5])]; check('explanation export scope',19740,len(export),0); check('empty explanations N2/5',274,export.text.str.strip().eq('').sum(),0)
    result=pd.DataFrame(rows); table(result,'00_reproduction'); passed=bool(result['pass'].all())
    (HERE/'results/00_gate.json').write_text(json.dumps(dict(passed=passed,failures=result.loc[~result['pass'],'quantity'].tolist(),seed=SEED,source_sha256=SOURCE_HASH),indent=2))
    failures=result[~result['pass']]
    section('Task 0 — reproduction gate',f'Gate: **{"PASS" if passed else "FAIL — STOP"}**. {len(result)-len(failures)}/{len(result)} anchors pass. All estimates recomputed from actions and original reliability flags. Cooperation tolerance 0.01 pp, D 0.05 pp, counts exact; other values checked against their published rounding intervals. Welch episode sample variances use ddof=1; eight-cell weights and separate-mode OLS agree.\n\n'+(failures.to_string(index=False) if len(failures) else f'Llama D={effects.iloc[0].D:.5f} pp; Q(5)={h["Q"]:.8f}, p={h["p"]:.8f}, I²={h["I2"]:.5f}%.')+'\n\nNo Tasks 1–4 may run following a failed gate without explicit acceptance of discrepancies. All subsequent work remains exploratory.')
    print(result.to_string(index=False)); print('GATE',passed)
    if not passed: raise SystemExit(2)

if __name__=='__main__': main()
