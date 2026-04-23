from pathlib import Path
import json

match_dir = Path('src/results/causal_theta/ct__N5__M5__th0p10-0p25-0p50-0p75__p0.05__lam0.25__seeds1__t50__20260423_134651/dsv32-llama31-gptoss20-qwen3235-gemma327__hb61c51__n5__m5__th0p10__p0.05__lam0.25__s101')
print('has __th:', '__th' in match_dir.name)
print('is_dir:', match_dir.is_dir())

agents_root = match_dir / 'agents'
for agent_dir in agents_root.iterdir():
    tp = agent_dir / 'agent_traces.jsonl'
    sz = tp.stat().st_size if tp.exists() else 0
    print('agent_dir=%s, trace_exists=%s, size=%d' % (agent_dir.name, tp.exists(), sz))
    if tp.exists():
        with open(tp, encoding='utf-8') as f:
            line = f.readline().strip()
        if line:
            d = json.loads(line)
            print('  round=%s action=%s B_eff=%s' % (d.get('round'), d.get('decision',{}).get('a'), d.get('B_eff')))

# Now test extract_episode directly
import sys
sys.path.insert(0, '.')
from extract_all import extract_episode
recs = extract_episode(match_dir, 'causal_theta', 'theta=0.1')
print('Records returned:', len(recs))
if recs:
    print('Sample:', {k: recs[0][k] for k in ['model','round','action','true_coop','theta','B_eff']})
