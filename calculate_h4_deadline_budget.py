"""Review-only budget estimates. Does not generate execution plans or run jobs."""
import collections
import json
import math
from pathlib import Path
from calculate_h4_per_model_power import model_power

OUT = Path('results/analysis/h4_deadline_budget_20261007')
HEAVY = ['llama-3.1-8b-instruct', 'gemma-3-12b-it', 'deepseek-r1-distill-qwen-14b']
GPT = 'gpt-oss-120b'


def target(record, margin):
    n = 2
    while model_power(.15, margin * record['q_m'], record['welch_df_factor'], n) < .9:
        n += 1
    return n


def allocate(targets, rates, budget):
    fractions = {0., 1.}
    for name in HEAVY:
        fractions.update(n / targets[name] for n in range(6, targets[name] + 1))
    best = None
    for fraction in sorted(fractions):
        counts = {name: max(6, min(targets[name], math.floor(fraction * targets[name] + 1e-10))) for name in HEAVY}
        if sum(counts[name] * rates[name] for name in HEAVY) <= budget:
            best = counts
    return best


def calculate():
    base = Path('results/analysis/h4_amended_power_20261007')
    power = json.loads((base / 'per_model_power.json').read_text())
    costs = json.loads((base / 'compute_cost.json').read_text())
    old = json.loads(Path('results/analysis/main_dry_run_review_20261007/run_plan.json').read_text())
    smoke = collections.defaultdict(float)
    for cell in old['cells']:
        smoke[cell['model']] += cell['estimated_gpu_hours'] / 40
    records = {r['model']: r for r in power['models']}
    rates = {name: max(smoke[name], costs['per_model_main_hours_from_completed_pilot_episodes'][name] / 40) for name in records}
    targets = {name: target(r, 1.25) for name, r in records.items()}
    cheap = {name: targets[name] for name in records if name not in HEAVY}
    cheap_h100 = sum(rates[name] * n for name, n in cheap.items() if name != GPT)
    scenarios = []
    for collection_hours in (84, 96):
        productive = collection_hours - 6
        counts = allocate(targets, rates, 2 * productive - cheap_h100)
        if counts is None:
            raise ValueError('Minimum six fresh heavy seeds cannot fit conditional capacity')
        counts.update(cheap)
        rows = []
        for name, record in records.items():
            n = counts[name]
            rows.append(dict(model=name, target=targets[name], fresh_seeds_per_cell=n, pilot_credit=0,
                             new_episodes=8*n, gpu_hours=n*rates[name],
                             estimated_power=model_power(.15, 1.25*record['q_m'], record['welch_df_factor'], n)))
        h100 = sum(r['gpu_hours'] for r in rows if r['model'] != GPT)
        h200 = rates[GPT] * counts[GPT]
        scenarios.append(dict(collection_window_hours=collection_hours, operational_reserve_hours=6,
                              productive_hours=productive, conditional_h100_slots=2, conditional_h200_slots=1,
                              conditional_gpu_hour_ceiling=3*productive, rows=rows,
                              episodes=sum(r['new_episodes'] for r in rows), gpu_hours=h100+h200,
                              capacity_lower_bound_hours=max(h100/2, h200),
                              unsplit_model_bottleneck_hours=max(r['gpu_hours'] for r in rows),
                              h100_work_hours=h100, h200_work_hours=h200,
                              required_deepseek_sharding=True, approved=False))
    minimum = {name: 6 if name in HEAVY else targets[name] for name in records}
    return dict(status='CONDITIONAL_REVIEW_ONLY_NOT_A_DRY_RUN', target_effect=.15, variance_margin=1.25,
                alpha=.05, target_power=.9, confirmed_new_validated_slots_now=0,
                pilot_credit=0, rates_gpu_hours_per_balanced_seed=rates,
                targets=[dict(model=name, raw=target(r, 1.), margin=targets[name], q_m=r['q_m'],
                              welch_df_factor=r['welch_df_factor']) for name, r in records.items()],
                minimum_fresh_gpu_hours=sum(minimum[name]*rates[name] for name in records),
                minimum_h100_hours=sum(minimum[name]*rates[name] for name in records if name != GPT),
                full_targets_gpu_hours=sum(targets[name]*rates[name] for name in records),
                scenarios=scenarios, execution_authorized=False)


if __name__ == '__main__':
    result = calculate()
    with (OUT / 'conditional_deadline_budget.json').open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result, indent=2))
