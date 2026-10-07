"""Review-only per-model seed/cost estimates; never creates dry-runs or approvals."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
from calculate_h4_per_model_power import model_power


def margin_target(q, df_factor, multiplier=1.25):
    n=2
    while model_power(.10,multiplier*q,df_factor,n)<.9:
        n+=1
    return n


def estimate(power_path, cost_path, existing_plan_path):
    power=json.loads(Path(power_path).read_text())
    cost=json.loads(Path(cost_path).read_text())
    plan=json.loads(Path(existing_plan_path).read_text())
    smoke=collections.defaultdict(float)
    for c in plan['cells']:smoke[c['model']]+=c['estimated_gpu_hours']
    rows=[]
    for m in power['models']:
        target=margin_target(m['q_m'],m['welch_df_factor'])
        gpt=m['model']=='gpt-oss-120b'
        rows.append({'model':m['model'],'raw_required':m['requirements'][1]['seeds_per_cell'],
                     'variance_margin_multiplier':1.25,'margin_adjusted_target':target,
                     'power_at_adjusted_target':model_power(.10,1.25*m['q_m'],m['welch_df_factor'],target),
                     'pilot_seeds_available_per_cell':5,'pilot_credit_applied':0,
                     'provisional_credit_if_all_conditions_match':5,
                     'new_seeds_if_full_credit_approved':max(0,target-5),
                     'new_episodes_if_full_credit_approved':8*max(0,target-5),
                     'new_seeds_if_no_credit':target,'new_episodes_if_no_credit':8*target,
                     'reuse_status':'Source compatible; serving and adaptive-inference review pending' if gpt
                         else 'Original parser differs from main v2; selected pilot credit blocked pending decision',
                     'pilot_parser':'final_decision_v2' if gpt else 'original extraction',
                     'proposed_main_parser':'final_decision_v2'})
    scenarios=[]
    for label in ['all_pilot_credit_counterfactual','gpt_only_credit_if_approved','no_pilot_credit']:
        counts={r['model']:max(0,r['margin_adjusted_target']-5) if label=='all_pilot_credit_counterfactual'
                else 0 if label=='gpt_only_credit_if_approved' and r['model']=='gpt-oss-120b'
                else r['margin_adjusted_target'] for r in rows}
        baselines={'completed_pilot_timing':cost['per_model_main_hours_from_completed_pilot_episodes'],
                   'existing_dry_run_smoke_timing':smoke}
        timings={}
        for basis,baseline in baselines.items():
            hours={model:baseline[model]*n/40 for model,n in counts.items()}
            timings[basis]={'per_model_new_gpu_hours':hours,'total_new_gpu_hours':sum(hours.values()),
                           'ideal_parallel_days':max(hours.values())/24,'serial_days':sum(hours.values())/24,
                           'reduction_percent_vs_same_basis_uniform40':100*(1-sum(hours.values())/sum(baseline.values()))}
        scenarios.append({'label':label,'approved':False,'new_seeds_by_model':counts,
                          'total_new_episodes':8*sum(counts.values()),'timings':timings})
    return {'status':'REVIEW_ONLY_NO_REUSE_CREDIT_APPLIED_NO_DRY_RUN',
            'sources':{str(Path(p).resolve()):hashlib.sha256(Path(p).read_bytes()).hexdigest()
                       for p in (power_path,cost_path,existing_plan_path)},
            'target_effect':.10,'target_power_per_model':.9,'rows':rows,'scenarios':scenarios,
            'retention_rule':'If approved, retain all five pilot seeds per cell even where target is two; no post-hoc seed selection.',
            'execution_authorized':False,'new_dry_run_generated':False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--power',type=Path,required=True);p.add_argument('--cost',type=Path,required=True)
    p.add_argument('--existing-plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=estimate(a.power,a.cost,a.existing_plan)
    a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'optimization_estimates.json').open('x') as f:json.dump(r,f,indent=2);f.write('\n')
    for row in r['rows']:print(row['model'],row['raw_required'],row['margin_adjusted_target'])
    for s in r['scenarios']:print(s['label'],s['total_new_episodes'],s['timings'])


if __name__=='__main__':main()
