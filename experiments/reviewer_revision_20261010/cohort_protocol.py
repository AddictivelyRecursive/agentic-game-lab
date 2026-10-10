"""Prospective four-arm message construction and immutable analysis constants."""
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'src'))
from AI_Agent.agent.pipeline import prepare_context,build_payload,rank_actions_by_eu

def messages_for(turn,arm,repair=None):
    if arm not in ['F','S','EV-canonical','EV-sorted']: raise ValueError('Unknown arm')
    state=prepare_context(dict(raw_input=turn))
    payload=build_payload(state,dict(reasoning_mode='structured' if arm=='S' else 'free_form',memory_mode='history_only',predict_opponents=False))
    if arm.startswith('EV-'):
        scores=rank_actions_by_eu(state)
        if arm=='EV-canonical': scores=sorted(scores,key=lambda r:r['action'])
        payload['expected_utility_ranking']=scores
    prompts=ROOT/'src/AI_Agent/prompts'
    if repair is not None:
        payload.update(task='Repair the invalid output without changing a valid action unnecessarily.',validation_error=repair['error'],previous_invalid_output=repair['output'])
        policy=(prompts/'repair_system.txt').read_text(encoding='utf-8')
    elif arm.startswith('EV-'):
        policy=(HERE/'neutral_policy.txt').read_text(encoding='utf-8')
    else:
        policy=(prompts/('decision_policy_structured.txt' if arm=='S' else 'decision_policy_system.txt')).read_text(encoding='utf-8')
    return [dict(role='system',content=(prompts/'base_system.txt').read_text(encoding='utf-8')+'\n'+policy),dict(role='user',content=json.dumps(payload,ensure_ascii=False,indent=2))]
