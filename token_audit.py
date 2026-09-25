import sys, json, statistics, tempfile
from pathlib import Path
from dataclasses import replace
ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT/'.token-audit-deps'), str(ROOT/'src')]
import tiktoken
from game_engine.env.types import EnvConfig, PayoffConfig, DriftConfig, StreakConfig, ObservationConfig
from game_engine.env.simulator import GameSimulator
from game_engine.agents.llm_wrapper import LLMWrapperAgent

encs = {n:tiktoken.get_encoding(n) for n in ('cl100k_base','o200k_base')}
calls=[]
class Capture:
    def generate(self, system_prompt, user_prompt):
        calls.append((system_prompt,user_prompt))
        turn=json.loads(user_prompt)['turn']
        return json.dumps({'a':(turn['round']+turn['agent_id']) % turn['game_parameters']['M'], 'reason':'Offline token audit.', 'expectation':'Unknown.'})

report={}
for m in (2,5):
    calls.clear()
    cfg=EnvConfig(N=2,M=m,T=50,p_perception=.05,payoff=PayoffConfig(12.,8.,0.,9.,15.),drift=DriftConfig(8,.35,.55),streak=StreakConfig(.6,.25,4.),obs=ObservationConfig(10,10),seed=101)
    with tempfile.TemporaryDirectory() as d:
        agents=[LLMWrapperAgent(name=f'p{i}',agent_id=i,env_cfg=cfg,backend='dummy',prompt_dir=str(ROOT/'src/AI_Agent/prompts'),output_dir=str(Path(d)/str(i))) for i in range(2)]
        for a in agents: a.llm.llm_client=Capture()
        GameSimulator(cfg).run_episode(agents)
    report[f'M={m}']={}
    for name, enc in encs.items():
        counts=[sum(len(enc.encode(s)) for s in pair) for pair in calls]
        report[f'M={m}'][name]={'calls':len(counts),'system_tokens':len(enc.encode(calls[0][0])),'first_round_both':sum(counts[:2]),'round_11_both':sum(counts[20:22]),'average_round_both':sum(counts)/50,'input_total_50_rounds':sum(counts)}

outputs=[]
files=0
for run in (ROOT/'src/results/llm_vs_baseline').glob('*202609*'):
    for path in run.rglob('agent_traces.jsonl'):
        files+=1
        for line in path.read_text(encoding='utf-8').splitlines():
            if not line.startswith('{'): continue
            s=json.loads(line)
            raw=s.get('llm_raw_outputs',{}).get('N6')
            if raw and s.get('valid_model_decision'):
                outputs.append(raw)
report['saved_valid_outputs']={'files_checked':files,'responses':len(outputs)}
for name,enc in encs.items():
    ns=sorted(len(enc.encode(s)) for s in outputs)
    if ns: report['saved_valid_outputs'][name]={'mean':statistics.mean(ns),'median':statistics.median(ns),'min':min(ns),'max':max(ns),'p90':ns[int(.9*(len(ns)-1))]}
print(json.dumps(report,indent=2))
(ROOT/'token_audit_results.json').write_text(json.dumps(report,indent=2))
