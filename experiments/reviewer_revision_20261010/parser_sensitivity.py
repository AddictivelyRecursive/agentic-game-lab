"""POST-HOC / EXPLORATORY: v3 syntax sensitivity on fixed saved histories."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'analysis'))
from common import DEFAULT_INPUT,SOURCE_HASH
from frozen_parser_v2 import extract_decision_json as v2,validate_decision
from parser_v3 import extract_decision_json as v3,VERSION
from diagnose_saved import table

def action(parser,text):
    try:
        result=validate_decision(dict(decision=parser(text),M=5))
        return result['decision']['a'] if result['is_valid'] else None
    except (ValueError,TypeError): return None

def main():
    assert hashlib.sha256(DEFAULT_INPUT.read_bytes()).hexdigest()==SOURCE_HASH
    rows=[]
    for line in DEFAULT_INPUT.open(encoding='utf-8'):
        ep=json.loads(line)
        if 'decisions' not in ep or ep['model'] not in ['llama-3.1-8b-instruct','gpt-oss-120b']: continue
        actual={r['t']:r['true_actions'] for r in ep['rounds']}
        for d in ep['decisions']:
            text=d['model_calls'][0].get('output')
            a2=action(v2,text) if isinstance(text,str) else None
            a3=action(v3,text) if isinstance(text,str) else None
            if a2 is not None and a2!=a3: raise ValueError('v2-valid action changed')
            rows.append(dict(model=ep['model'],mode=ep['reasoning_mode'],N=ep['N'],episode_id=ep['episode_id'],round=d['round'],seat=d['seat'],v2_valid=a2 is not None,v3_valid=a3 is not None,recovered=a2 is None and a3 is not None,v2_action=a2,v3_action=a3,executed_action=actual[d['round']][d['seat']],v2_c=1-a2/4 if a2 is not None else np.nan,v3_c=1-a3/4 if a3 is not None else np.nan,executed_c=1-actual[d['round']][d['seat']]/4))
    dd=pd.DataFrame(rows); out=[]
    for keys,g in dd.groupby(['model','mode','N']):
        recovered=g[g.recovered]
        out.append(dict(zip(['model','mode','N'],keys),decisions=len(g),v2_valid=int(g.v2_valid.sum()),v3_valid=int(g.v3_valid.sum()),recovered=int(g.recovered.sum()),v2_valid_pct=100*g.v2_valid.mean(),v3_valid_pct=100*g.v3_valid.mean(),recovered_action_differs_from_executed=int((recovered.v3_action!=recovered.executed_action).sum())))
    table(pd.DataFrame(out),'00b_parser_v3_sensitivity')
    dd.to_csv(HERE/'results/00b_parser_v3_decisions.csv',index=False)
    conditional=[]; rng=np.random.default_rng(20261010)
    for (model,mode),g in dd.groupby(['model','mode']):
        for phase,subset in [('all',g),('early',g[g['round']<=5]),('late',g[g['round']>=26])]:
            for metric in ['v2_c','v3_c','executed_c']:
                episode_means=subset.groupby(['N','episode_id'])[metric].mean()
                cells=[episode_means.loc[n].dropna().to_numpy() for n in [2,3,4,5]]
                available=all(len(x)>0 for x in cells)
                if available:
                    mean=100*np.mean([x.mean() for x in cells]); draws=100*np.mean([x[rng.integers(0,len(x),(5000,len(x)))].mean(axis=1) for x in cells],axis=0); lo,hi=np.quantile(draws,[.025,.975])
                else: mean=lo=hi=np.nan
                conditional.append(dict(model=model,mode=mode,phase=phase,estimator=metric,cooperation_pct=mean,low_pct=lo,high_pct=hi,eligible_episodes_by_N=';'.join(f'{n}:{len(x)}' for n,x in zip([2,3,4,5],cells)),status='conditional on a parseable first attempt; fixed observed histories; selection biased' if metric!='executed_c' else 'historical executed actions; unchanged primary trajectories'))
    table(pd.DataFrame(conditional),'00b_parser_conditional_outcomes')
    text=['# Parser v3 saved-log sensitivity — POST-HOC / EXPLORATORY','',f'Protocol candidate `{VERSION}`. v3 accepts only complete decision objects, additionally permitting literal newline, carriage-return and tab in strings. It does not complete truncated JSON, infer missing actions, or select among multiple decisions. Every v2-valid action is verified unchanged.','', 'These are re-extracted first attempts on unchanged observed histories, not fresh trajectories under v3. Recovered actions may differ from historically executed repaired actions. Neither primary outcomes nor repair/fallback flags are overwritten.','']
    for model,g in dd.groupby('model'):
        failures=int((~g.v2_valid).sum()); recovered=int(g.recovered.sum())
        text += [f'## {model}','',f'{recovered}/{failures} first-attempt failures recovered. v2 validity {100*g.v2_valid.mean():.2f}%; v3 validity {100*g.v3_valid.mean():.2f}%.','']
        if recovered/failures>.5:
            text += ['Decision: adopt the explicitly versioned v3 candidate in a separately registered new cohort, reporting prospective reliability during its first balanced block and retaining those episodes. The dominant literal-whitespace problem is fixable without inventing an action. This completes the CPU-only parser branch of Stage 0b; no extra Llama model calls were made.','']
        else:
            text += ['Decision: v3 does not resolve the dominant failures. Do not enroll the full cohort yet; inspect missing-response errors and run a fully instrumented budget/transport diagnostic under a new protocol. Saved outputs alone cannot certify token truncation.','']
    text += ['## Conditional cooperation sensitivity','', '00b_parser_conditional_outcomes.csv reports v2-eligible and v3-eligible first-attempt cooperation alongside unchanged executed cooperation, with early/late summaries and 5,000 episode-bootstrap intervals within N cells. These are conditional estimates on the original observed histories, not an intervention rerunning those histories. Cells with no eligible episodes remain not estimable; they are not dropped from equal-N aggregation. Two old GPT-OSS seeds per cell make its bootstrap uncertainty particularly weak.']
    (HERE/'PARSER_V3_SENSITIVITY.md').write_text('\n'.join(text),encoding='utf-8')
    print('\n'.join(text))

if __name__=='__main__': main()
