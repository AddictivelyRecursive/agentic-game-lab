"""POST-HOC / EXPLORATORY Stage 0b; saved first attempts only, no inference."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'analysis'))
from common import DEFAULT_INPUT,SOURCE_HASH,SEED
from frozen_parser_v2 import extract_decision_json,validate_decision

CATEGORIES=['malformed_json_or_wrapper','missing_field','invalid_action','truncation_or_empty','other']

def classify(call):
    text=call.get('output')
    if call.get('error') and not isinstance(text,str):
        return 'other','logged_transport_or_generation_error',False,False
    if not isinstance(text,str) or not text.strip():
        return 'truncation_or_empty','empty_output_not_proven_truncation',False,False
    try:
        obj=extract_decision_json(text)
    except (ValueError,TypeError) as exc:
        if '<think>' in text and '</think>' not in text:
            return 'truncation_or_empty','unterminated_think_not_proven_token_limit',False,False
        if call.get('finish_reason')=='length':
            return 'truncation_or_empty','recorded_length_finish',False,False
        # v2 already accepts markdown fences and surrounding prose. Inspect only
        # complete JSON values; never infer a submitted action from prose.
        final=text.rsplit('</think>',1)[-1]
        objects=[]; decoder=json.JSONDecoder(); offset=0
        while offset<len(final):
            start=final.find('{',offset)
            if start<0: break
            try: value,end=decoder.raw_decode(final,start)
            except json.JSONDecodeError: offset=start+1; continue
            if isinstance(value,dict): objects.append(value)
            offset=end
        decisions=[o for o in objects if 'a' in o]
        if len(decisions)>1:
            return 'malformed_json_or_wrapper','multiple_decision_objects_ambiguous',False,False
        if objects and not decisions and '"a"' not in final:
            return 'missing_field','complete_json_without_a',False,False
        return 'malformed_json_or_wrapper','no_complete_decision_json',False,False
    result=validate_decision(dict(decision=obj,M=5))
    if not result['is_valid']:
        return 'invalid_action',result['validation_error'],True,False
    return 'valid','v2_valid',True,True

def table(df,name):
    out=HERE/'results'; (out/'latex').mkdir(parents=True,exist_ok=True)
    df.to_csv(out/(name+'.csv'),index=False)
    (out/'latex'/(name+'.tex')).write_text('% POST-HOC / EXPLORATORY\n'+df.to_latex(index=False,escape=True,float_format=lambda x:f'{x:.2f}'),encoding='utf-8')

def main():
    if hashlib.sha256(DEFAULT_INPUT.read_bytes()).hexdigest()!=SOURCE_HASH: raise ValueError('Source changed')
    rows=[]; call_keys=set()
    for line in DEFAULT_INPUT.open(encoding='utf-8'):
        ep=json.loads(line)
        if 'decisions' not in ep or ep['model'] not in ['llama-3.1-8b-instruct','gpt-oss-120b']: continue
        for d in ep['decisions']:
            calls=d['model_calls']
            if not calls: raise ValueError('Missing first attempt')
            call=calls[0]; call_keys.update(call)
            cat,detail,parsed,valid=classify(call)
            text=call.get('output')
            rows.append(dict(model=ep['model'],mode=ep['reasoning_mode'],N=ep['N'],episode_id=ep['episode_id'],seed=ep['seed'],round=d['round'],seat=d['seat'],category=cat,detail=detail,first_parse=parsed,first_valid=valid,repair=d['repair_used'],fallback=d['fallback_used'],historical_first_parse=d['first_parse_ok'],output_sha256=hashlib.sha256((text or '').encode()).hexdigest(),output=text or '',logged_error=str(call.get('error') or '')))
    df=pd.DataFrame(rows)
    (HERE/'results').mkdir(exist_ok=True)
    df.to_csv(HERE/'results/00b_first_attempt_decisions.csv',index=False)
    # Cluster resampling within each N cell. Aggregate mode summaries give four
    # cells equal weight; conditional failure composition uses the pooled count
    # ratio because cells have equal decision exposure only within N.
    rng=np.random.default_rng(SEED); summaries=[]
    def summarize(sub,model,mode,n_label):
        cells=[]
        for _,cell in sub.groupby('N'):
            episodes=[]
            for _,g in cell.groupby('episode_id'):
                episodes.append([len(g),int(g.first_parse.sum()),int(g.first_valid.sum())]+[int((g.category==cat).sum()) for cat in CATEGORIES])
            cells.append(np.asarray(episodes,dtype=float))
        observed=np.sum([c.sum(axis=0) for c in cells],axis=0)
        draws=np.sum([c[rng.integers(0,len(c),(5000,len(c)))].sum(axis=1) for c in cells],axis=0)
        for k,label in [(1,'first_parse'),(2,'first_valid')]+[(i+3,cat) for i,cat in enumerate(CATEGORIES)]:
            values=100*draws[:,k]/draws[:,0]; lo,hi=np.quantile(values,[.025,.975])
            summaries.append(dict(model=model,mode=mode,N=n_label,metric=label,decisions=int(observed[0]),episodes=sum(len(c) for c in cells),count=int(observed[k]),rate_pct=100*observed[k]/observed[0],low_pct=lo,high_pct=hi,share_of_failures_pct=100*observed[k]/(observed[0]-observed[2]) if k>=3 and observed[0]>observed[2] else np.nan,interval='5000 episode bootstrap draws within N cells; pooled decision-rate ratio'))
    for (model,mode,n),sub in df.groupby(['model','mode','N']): summarize(sub,model,mode,str(n))
    for (model,mode),sub in df.groupby(['model','mode']): summarize(sub,model,mode,'all')
    table(pd.DataFrame(summaries),'00b_failure_taxonomy')
    details=df[df.category!='valid'].groupby(['model','mode','category','detail']).size().rename('count').reset_index()
    table(details,'00b_failure_details')
    memo=['# Stage 0b decision memo — POST-HOC / EXPLORATORY','',f'Source SHA256: `{SOURCE_HASH}`. All first attempts for Llama and GPT-OSS were inspected. Seed {SEED}; 5,000 episode-cluster bootstrap draws. No new inference.','', 'Failure categories are mutually exclusive. A logged error without a saved response is classified as other, not an invented empty response. Errors attached to an available output are classified from that output. Unterminated thinking and empty output are identifiable, but the saved call schema lacks token usage and finish reasons; a token-limit diagnosis is therefore unconfirmed. Incomplete JSON is classified as malformed, not silently called truncation. Nested complete objects inside an incomplete action object do not establish a missing action field. v2 already accepts markdown fences and surrounding prose.','']
    decisions=[]
    for model,sub in df.groupby('model'):
        failed=sub[sub.category!='valid']; f=len(failed)
        format_count=int(failed.category.isin(['malformed_json_or_wrapper','missing_field']).sum())
        generation_count=int(failed.category.isin(['invalid_action','truncation_or_empty']).sum())
        ambiguous=int((failed.detail=='multiple_decision_objects_ambiguous').sum())
        if f and format_count/f>.5:
            decision='Parser v3 investigation warranted by the prespecified majority rule; do not enroll until a conservative extractor is specified and saved-log sensitivity is complete. Ambiguous multiple actions must remain failures.'
        elif f and generation_count/f>.5:
            decision='Generation-side diagnostic required before enrollment; changing budgets, schema or prompt requires a new protocol ID. No parser relaxation is justified by this majority rule.'
        else:
            decision='Keep v2 provisionally; no format or generation category exceeds half of failures. Diagnose logged transport errors separately and preserve the repair path. Full-cohort enrollment remains conditional on the documented readiness screening.'
        decisions.append(dict(model=model,failed=f,format_related=format_count,generation_related=generation_count,multiple_objects=ambiguous,decision=decision))
        memo += [f'## {model}', '',f'{f} first-attempt failures; {format_count} format/missing-field, {generation_count} invalid-action/empty/unterminated-thinking; {ambiguous} ambiguous multiple-object outputs.', '',decision,'']
        for mode,g in sub.groupby('mode'):
            memo.append(f'- {mode}: parse {100*g.first_parse.mean():.2f}%; valid {100*g.first_valid.mean():.2f}%; repair {100*g.repair.mean():.2f}%; fallback {100*g.fallback.mean():.2f}%.')
        memo.append('')
    memo += ['## Follow-up decision','', 'The completed conservative-parser investigation and final enrollment recommendations are in PARSER_V3_SENSITIVITY.md. They supersede the initial investigation branches above; the historical data and primary outcomes remain unchanged.','', '## Caveats','', 'Rates describe the original tested pipeline. Bootstrap intervals cannot correct two-seed uncertainty for GPT-OSS. Diagnostic categories do not identify a model’s intent. Any future cohort has a new protocol ID and is analyzed independently. No downstream cooperation values were used to assign categories or make the decision.','',f'Observed saved first-call keys: {sorted(call_keys)}.']
    (HERE/'STAGE_0B_MEMO.md').write_text('\n'.join(memo)+'\n',encoding='utf-8')
    (HERE/'results/00b_decision.json').write_text(json.dumps(decisions,indent=2),encoding='utf-8')
    print('\n'.join(memo))

if __name__=='__main__': main()
