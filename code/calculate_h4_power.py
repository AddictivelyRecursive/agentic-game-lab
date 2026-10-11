"""Auditable, conditional H4 episode-mean power; never creates a main power approval."""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

from scipy.stats import norm


def two_sided_power(effect, variance, seeds, alpha=.05):
    critical = norm.ppf(1 - alpha / 2)
    noncentrality = abs(effect) / math.sqrt(variance / seeds)
    return float(norm.sf(critical - noncentrality) + norm.cdf(-critical - noncentrality))


def required_seeds(effect, variance, target=.9):
    high = 1
    while two_sided_power(effect, variance, high) < target:
        high *= 2
    low = 1
    while low < high:
        middle = (low + high) // 2
        if two_sided_power(effect, variance, middle) >= target:
            high = middle
        else:
            low = middle + 1
    return low


def calculate(source):
    raw = Path(source).read_bytes()
    data = json.loads(raw)
    models, cells = data['models'], data['cells']
    if len(models) != 6 or len(set(models)) != 6:
        raise ValueError('Expected six models')
    keys = [(c['model'], c['reasoning_mode'], c['N']) for c in cells]
    expected = set(itertools.product(models, ('structured', 'free_form'), (2, 3, 4, 5)))
    if len(keys) != 48 or set(keys) != expected:
        raise ValueError('Expected unique complete 48-cell grid')
    if data['rounds_per_episode'] != 30 or data['variance_ddof'] != 1:
        raise ValueError('Expected 30-round episode sample variances')
    for c in cells:
        if c['completed'] != 5 or c['target'] != 5 or c['failed'] != 0:
            raise ValueError('Incomplete pilot cell')
        if not math.isfinite(c['variance']) or c['variance'] < 0:
            raise ValueError('Invalid empirical variance')
    # Sxx = sum((N-3.5)^2) = 5. Average slopes over six models.
    # Endpoint interaction D = 3*beta_interaction; weight = +/- (N-3.5)/10.
    inputs = []
    for c in cells:
        weight = (1 if c['reasoning_mode'] == 'free_form' else -1) * (c['N'] - 3.5) / 10
        inputs.append(dict(c, endpoint_contrast_weight=weight,
                           variance_contribution_at_one_seed=weight**2 * c['variance']))
    empirical_variance = math.fsum(c['variance_contribution_at_one_seed'] for c in inputs)
    maximum = max(c['variance'] for c in cells)
    if empirical_variance <= 0:
        raise ValueError('No positive empirical variance')
    working_variance = math.fsum(c['endpoint_contrast_weight']**2 * maximum for c in inputs)
    # Shared random model intercept and shared N slope cancel in this balanced contrast.
    cancellation = []
    for model in models:
        subset = [c for c in inputs if c['model'] == model]
        intercept = math.fsum(c['endpoint_contrast_weight'] for c in subset)
        slope = math.fsum(c['N'] * c['endpoint_contrast_weight'] for c in subset)
        assert abs(intercept) < 1e-12 and abs(slope) < 1e-12
        cancellation.append({'model': model, 'intercept_weight_sum': intercept, 'N_slope_weight_sum': slope})
    scenarios = []
    for label, variance in [('actual_cell_variances', empirical_variance),
                            ('common_max_observed_variance', working_variance)]:
        table = []
        for effect in (.05, .10, .15):
            n = required_seeds(effect, variance)
            table.append({'endpoint_interaction': effect, 'slope_interaction_per_player': effect / 3,
                          'seeds_per_cell': n, 'total_episodes': 48*n,
                          'power_at_n': two_sided_power(effect, variance, n),
                          'power_at_n_minus_one': two_sided_power(effect, variance, n-1) if n > 1 else None})
        assert all(a['seeds_per_cell'] >= b['seeds_per_cell'] for a,b in zip(table,table[1:]))
        scenarios.append({'variance_rule': label, 'endpoint_contrast_variance_at_one_seed': variance, 'table': table})
    return {'status': 'CONDITIONAL_APPROXIMATION_NOT_APPROVED_FOR_MAIN', 'source': str(Path(source).resolve()),
            'source_sha256': hashlib.sha256(raw).hexdigest(), 'target_power': .9, 'alpha_two_sided': .05,
            'raw_zero_variance_cells': sum(c['variance'] == 0 for c in cells),
            'total_cells': 48, 'raw_variances_modified': False,
            'maximum_observed_cell_variance': maximum,
            'maximum_variance_source_cells': [{k:c[k] for k in ('model','reasoning_mode','N','variance')} for c in cells if c['variance']==maximum],
            'episode_ICC': None, 'episode_random_intercept_variance': None,
            'ICC_status': 'Not present/identifiable from episode-mean variance summaries; no assumed ICC used',
            'simulation_replicates': 0, 'Monte_Carlo_error': None,
            'method': 'Closed-form normal plug-in power of equal-weight endpoint difference-of-slopes contrast',
            'model_effect_cancellation': cancellation, 'scenarios': scenarios,
            'assumptions': ['Independent episodes conditional on model; same seed across cells is not treated as a paired repeated episode.',
                            'Linear N slopes and equal model weights; all four N levels contribute.',
                            'Normal approximation for the episode-mean contrast, with pilot variances treated as known.',
                            'The common-maximum scenario imposes a homoscedastic working variance equal to an actual pilot variance, not a confidence bound.',
                            'Shared model random intercept and shared N slope cancel in the balanced contrast; mode-specific random slopes do not and are not estimated here.',
                            'No seat or round is counted as an independent sample; episode-mean variances already include their aggregate covariance.',
                            'This does not calibrate the fitted MixedLM Wald standard error, model selection, six-group finite-sample behavior or variance-estimation uncertainty.'],
            'review_blocker': 'Requested episode random-intercept GLMM is not the existing Gaussian episode-mean/model-group LMM. An ICC cannot be read from this variance file.',
            'seeds_per_cell_finalized': None}, inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variance', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result, inputs = calculate(args.variance)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'h4_power_calculation.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    fields = ['model','reasoning_mode','N','completed','variance','uniform','endpoint_contrast_weight','variance_contribution_at_one_seed']
    with (args.output/'variance_inputs.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for c in inputs: writer.writerow({k:c[k] for k in fields})
    for scenario in result['scenarios']:
        print(scenario['variance_rule'], [(r['endpoint_interaction'],r['seeds_per_cell']) for r in scenario['table']])


if __name__ == '__main__':
    main()
