"""Six-model H4 variance-floor review; no study approvals or inference calls."""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path

from scipy.stats import norm, nct, t
from calculate_h4_power import calculate


def cluster_power(effect, within, floor, seeds=math.inf, df=None):
    variance = floor + (0 if math.isinf(seeds) else within / seeds)
    if variance == 0:
        return .05 if effect == 0 else 1.
    noncentrality = abs(effect) / math.sqrt(variance)
    if df is None:
        critical = norm.ppf(.975)
        return float(norm.sf(critical-noncentrality)+norm.cdf(-critical-noncentrality))
    critical = t.ppf(.975, df)
    return float(nct.sf(critical,df,noncentrality)+nct.cdf(-critical,df,noncentrality))


def cluster_required_seeds(effect, within, floor, df=None):
    if cluster_power(effect, within, floor, df=df) <= .9:
        return None
    low, high = 1, 1
    while cluster_power(effect, within, floor, high, df) < .9:
        high *= 2
    while low < high:
        middle = (low+high)//2
        if cluster_power(effect, within, floor, middle, df) >= .9:
            high = middle
        else:
            low = middle+1
    return low


def review(source):
    earlier, inputs = calculate(source)
    per_model = []
    for model in [x['model'] for x in earlier['model_effect_cancellation']]:
        cells = [c for c in inputs if c['model']==model]
        # Six times the pooled contrast weights gives each model's endpoint interaction.
        effect = math.fsum(6*c['endpoint_contrast_weight']*c['mean_cooperation'] for c in cells)
        within = math.fsum((6*c['endpoint_contrast_weight'])**2*c['variance'] for c in cells)
        lookup = {(c['reasoning_mode'],c['N']):c for c in cells}
        literal = ((lookup['free_form',5]['mean_cooperation']-lookup['free_form',2]['mean_cooperation'])
                   -(lookup['structured',5]['mean_cooperation']-lookup['structured',2]['mean_cooperation']))
        per_model.append({'model':model,'fitted_endpoint_interaction':effect,
                          'fitted_slope_interaction_per_player':effect/3,
                          'literal_endpoint_interaction_sensitivity':literal,
                          'within_effect_variance_coefficient':within,
                          'effect_estimation_variance_at_five_seeds':within/5})
    k=len(per_model)
    observed = statistics.variance(c['fitted_endpoint_interaction'] for c in per_model)
    noise = statistics.mean(c['effect_estimation_variance_at_five_seeds'] for c in per_model)
    adjusted = max(0, observed-noise)
    floor = observed/k
    a = earlier['scenarios'][0]['endpoint_contrast_variance_at_one_seed']
    assert math.isclose(a,math.fsum(c['within_effect_variance_coefficient'] for c in per_model)/k**2,rel_tol=1e-12)
    scenarios=[]
    # Four primary rows: old/no floor and corrected/floor, each with two within treatments.
    for corrected in (False,True):
        for old in earlier['scenarios']:
            within=old['endpoint_contrast_variance_at_one_seed']
            current_floor=floor if corrected else 0
            rows=[]
            for effect in (.05,.10,.15):
                count=cluster_required_seeds(effect,within,current_floor)
                rows.append({'effect':effect,'seeds_per_cell':count,
                             'status':'90% unattainable for any finite seed count' if count is None else 'conditional integer threshold',
                             'power_ceiling_normal':cluster_power(effect,within,current_floor),
                             'power_at_1000_seeds':cluster_power(effect,within,current_floor,1000),
                             'power_at_threshold':cluster_power(effect,within,current_floor,count) if count is not None else None})
            scenarios.append({'label':('two_component_' if corrected else 'previous_within_only_')+old['variance_rule'],
                              'within_pooled_contrast_coefficient':within,'between_pooled_contrast_variance_floor':current_floor,
                              'table':rows})
    sensitivities=[]
    for treatment,between in [('observed_sample_variance',observed),('noise_adjusted_method_of_moments',adjusted)]:
        for old in earlier['scenarios']:
            within=old['endpoint_contrast_variance_at_one_seed']
            for df in (None,k-1):
                rows=[{'effect':e,'seeds_per_cell':cluster_required_seeds(e,within,between/k,df),
                       'power_ceiling':cluster_power(e,within,between/k,df=df)} for e in (.05,.10,.15)]
                sensitivities.append({'between_treatment':treatment,'within_treatment':old['variance_rule'],
                                      'test':'normal' if df is None else 'noncentral_t_df5_heuristic', 'df':df,'table':rows})
    return {'status':'RE_REVIEW_REQUIRED_NO_FINAL_SEED_COUNT','source':earlier['source'],
            'source_sha256':earlier['source_sha256'],'target_power':.9,'alpha_two_sided':.05,
            'cluster_count':k,'sample_variance_df':k-1,'per_model_estimates':per_model,
            'mean_fitted_endpoint_interaction':statistics.mean(c['fitted_endpoint_interaction'] for c in per_model),
            'observed_between_model_effect_sample_variance':observed,
            'mean_pilot_effect_estimation_noise_variance':noise,
            'noise_adjusted_latent_effect_variance_sensitivity':adjusted,
            'between_pooled_contrast_variance_floor':floor,
            'literal_endpoint_between_model_variance_sensitivity':statistics.variance(c['literal_endpoint_interaction_sensitivity'] for c in per_model),
            'primary_scenarios':scenarios,'sensitivity_scenarios':sensitivities,
            'episode_ICC':None,'simulation_replicates':0,'seeds_per_cell_finalized':None,
            'limitations':['Only six purposively selected models, not a random sample of a defined model population.',
                           'Sample variance has five degrees of freedom; heterogeneous/noisy estimated model effects are not exact independent Gaussian observations.',
                           'Using observed effect variance as latent variance conservatively double-counts some pilot estimation noise; adjustment shown separately.',
                           'Noise subtraction assumes independent estimation errors across models and cells; its uncertainty is not incorporated.',
                           'The current planned model has shared random N slope, but no random N-by-mode interaction slope; it does not implement this two-component H4 covariance.',
                           'Normal power treats estimated components as known. df=5 noncentral-t calculation is a small-cluster sensitivity heuristic, not exact heterogeneous MixedLM power.',
                           'No within-episode ICC is estimated; episode-mean variances contain aggregate seat/round covariance.',
                           'All raw variances and mean effects come from the pilot; normality, independence and variance transport are explicit planning assumptions.']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variance',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=review(args.variance)
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'h4_cluster_power.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with (args.output/'per_model_interactions.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(result['per_model_estimates'][0]));writer.writeheader()
        writer.writerows(result['per_model_estimates'])
    print('Observed between-model variance:',result['observed_between_model_effect_sample_variance'])
    for scenario in result['primary_scenarios']:
        print(scenario['label'],[(c['effect'],c['seeds_per_cell'],c['power_ceiling_normal']) for c in scenario['table']])


if __name__=='__main__':
    main()
