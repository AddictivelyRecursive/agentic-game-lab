import glob
import json
import os

# Sample episode_meta from each experiment type
experiments = {
    'causal_N_progressive': glob.glob('src/results/causal_N_progressive/**/episode_meta.json', recursive=True)[:3],
    'causal_M': glob.glob('src/results/causal_M/**/episode_meta.json', recursive=True)[:3],
    'causal_noise': glob.glob('src/results/causal_noise/**/episode_meta.json', recursive=True)[:3],
    'causal_theta': glob.glob('src/results/causal_theta/**/episode_meta.json', recursive=True)[:3],
}

for exp_type, files in experiments.items():
    print(f'\n=== {exp_type} === ({len(files)} samples)')
    for f in files:
        with open(f) as fp:
            meta = json.load(fp)
        config = meta.get('config', {})
        extra = meta.get('extra_meta', {})
        print(f'  File: {f}')
        print(f'  Config: {config}')
        print(f'  Extra: {extra}')
        print()

# Now sample a trace from causal_M to see what M means
print("\n=== SAMPLE TRACE causal_M ===")
traces = glob.glob('src/results/causal_M/**/agent_traces.jsonl', recursive=True)
if traces:
    with open(traces[0]) as fp:
        first_line = fp.readline()
    data = json.loads(first_line)
    # Print key fields
    print("Round:", data.get('round'))
    print("N:", data.get('N'))
    print("M:", data.get('M'))
    print("Action:", data.get('decision', {}).get('a'))
    print("B_eff:", data.get('B_eff'))
    print("Payoff:", data.get('payoff'))
    print("Candidates:", data.get('candidates', [])[:2])

# Sample a trace from causal_theta
print("\n=== SAMPLE TRACE causal_theta ===")
traces = glob.glob('src/results/causal_theta/**/agent_traces.jsonl', recursive=True)
if traces:
    with open(traces[0]) as fp:
        first_line = fp.readline()
    data = json.loads(first_line)
    print("Round:", data.get('round'))
    print("N:", data.get('N'))
    print("M:", data.get('M'))
    print("Action:", data.get('decision', {}).get('a'))
    print("streak_rule:", data.get('streak_rule', {}))
    print("B_eff:", data.get('B_eff'))

# Sample a trace from causal_noise
print("\n=== SAMPLE TRACE causal_noise ===")
traces = glob.glob('src/results/causal_noise/**/agent_traces.jsonl', recursive=True)
if traces:
    with open(traces[0]) as fp:
        first_line = fp.readline()
    data = json.loads(first_line)
    print("Round:", data.get('round'))
    print("p (noise):", data.get('p'))
    print("N:", data.get('N'))
    print("M:", data.get('M'))
    print("Action:", data.get('decision', {}).get('a'))
    print("noise_model:", data.get('noise_model'))
    print("B_eff:", data.get('B_eff'))
