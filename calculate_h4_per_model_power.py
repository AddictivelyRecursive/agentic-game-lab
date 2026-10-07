"""Amended named-model H4 planning: pilot q_m and Welch normal-data approximation."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from scipy.stats import nct, t
from calculate_h4_power import calculate


def model_power(effect, q, df_factor, seeds):
    df = df_factor * (seeds-1)
    critical = t.ppf(.975, df)
    noncentrality = abs(effect)/math.sqrt(q/seeds)
    # Symmetry avoids SciPy's occasional NaN in a tiny negative CDF tail.
    return float(nct.sf(critical,df,noncentrality)+nct.sf(critical,df,-noncentrality))


def calculate_per_model(source):
    earlier, inputs=calculate(source)
    rows=[]
    for record in earlier['model_effect_cancellation']:
        model=record['model']; cells=[c for c in inputs if c['model']==model]
        contributions=[(6*c['endpoint_contrast_weight'])**2*c['variance'] for c in cells]
        q=math.fsum(contributions)
        factor=q*q/math.fsum(x*x for x in contributions)
        table=[]
        for effect in (.05,.10,.15):
            n=2
            while model_power(effect,q,factor,n)<.9:
                n+=1
            table.append({'effect':effect,'seeds_per_cell':n,'power':model_power(effect,q,factor,n),
                          'power_at_previous_seed':model_power(effect,q,factor,n-1) if n>2 else None})
        rows.append({'model':model,'q_m':q,'welch_df_factor':factor,'requirements':table})
    uniform=max(r['requirements'][1]['seeds_per_cell'] for r in rows)
    for r in rows:r['power_at_uniform_10_point_target']=model_power(.10,r['q_m'],r['welch_df_factor'],uniform)
    return {'estimand':'six separate named-model endpoint slope interactions',
            'source':earlier['source'],'source_sha256':earlier['source_sha256'],
            'target_power':.9,'working_endpoint_effect':.10,'alpha_two_sided_per_model':.05,
            'method':'noncentral-t plug-in approximation with Welch-Satterthwaite df from eight cell variances',
            'assumptions':['Independent episode means conditional on model; normal episode sampling for t approximation.',
                           'Pilot cell variances transport to main with same horizon and fixed parser v2.',
                           'Pilot variances are plug-in estimates with only five seeds; uncertainty not integrated.',
                           'Power is marginal per-model, not simultaneous detection of all six or Holm-adjusted power.',
                           'Raw zero cell variances remain zero, not invented; none of the six q_m totals is zero.'],
            'models':rows,'uniform_seeds_per_cell':uniform,'total_episodes':48*uniform,
            'total_seat_decisions':5040*uniform,'amendment_path':'docs/H4_amendment_20261007.txt'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variance',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=calculate_per_model(a.variance);a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'per_model_power.json').write_text(json.dumps(r,indent=2)+'\n')
    print('Uniform seeds/cell:',r['uniform_seeds_per_cell'])
    for m in r['models']:print(m['model'],[x['seeds_per_cell'] for x in m['requirements']])


if __name__=='__main__':main()
