"""Offline safety/identification checks. No API calls or model inference."""
import json
from pathlib import Path
import tempfile
import unittest
from cohort_protocol import messages_for,HERE
from cohort_runner import Agent
from frozen_parser_v2 import extract_decision_json as v2
from parser_v3 import extract_decision_json as v3,validate_decision
from game_engine.env.types import EnvConfig,PayoffConfig,DriftConfig,StreakConfig
from game_engine.env.simulator import GameSimulator
import pandas as pd
from analyze_cohort import contrast

class ProtocolTests(unittest.TestCase):
    def test_registered_interaction_scale(self):
        rows=[]
        for n in [2,3,4,5]:
            for arm,mean in [('F',.1*(n-1)),('S',.1)]:
                for delta in [-.01,.01]: rows.append(dict(model='fixture',arm=arm,N=n,c=mean+delta))
        ep=pd.DataFrame(rows)
        self.assertAlmostEqual(contrast(ep,'fixture','F','S',True)['estimate_pp'],30.)
        self.assertAlmostEqual(contrast(ep,'fixture','F','S',False)['estimate_pp'],15.)

    def test_original_messages_and_ev_identification(self):
        bank=[json.loads(s) for s in (HERE/'request_bank.jsonl').read_text(encoding='utf-8').splitlines()]
        for state in bank:
            turn=json.loads(state['user'])['turn']; arm='S' if state['mode']=='structured' else 'F'
            self.assertEqual(messages_for(turn,arm),[dict(role='system',content=state['system']),dict(role='user',content=state['user'])])
            canon=messages_for(turn,'EV-canonical'); sorted_=messages_for(turn,'EV-sorted')
            self.assertEqual(canon[0],sorted_[0])
            c=json.loads(canon[1]['content']); s=json.loads(sorted_[1]['content'])
            cs=c.pop('expected_utility_ranking'); ss=s.pop('expected_utility_ranking')
            self.assertEqual(c,s)
            self.assertEqual(cs,sorted(ss,key=lambda r:r['action']))
            self.assertEqual([r['action'] for r in cs],list(range(5)))

    def test_v3_preserves_complete_values_and_rejects_unsafe_cases(self):
        text='```json\n{"a": 2, "reason": "line one\nline two"}\n```'
        with self.assertRaises(ValueError): v2(text)
        self.assertEqual(v3(text),{'a':2,'reason':'line one\nline two'})
        for text in ['{"a": 4, "reason": "unfinished','<think>{"a":4}', '{"a":0} {"a":4}', 'I choose action 3', '{"reason":"a=4"}']:
            with self.assertRaises(ValueError,msg=text): v3(text)
        for value in [True,5,-1,1.5,'bad']:
            self.assertFalse(validate_decision(dict(decision=v3(json.dumps({'a':value})),M=5))['is_valid'])

    def test_full_fallback_path_and_frozen_benefit(self):
        class Tokenizer:
            def apply_chat_template(self,*args,**kwargs): return [1,2,3]
        class Response:
            status_code=200
            def json(self): return dict(count=3,choices=[dict(finish_reason='stop',message=dict(content='invalid'))],usage=dict(completion_tokens=1))
            def raise_for_status(self): pass
        class Client:
            def post(self,*args,**kwargs): return Response()
        cfg=EnvConfig(N=2,M=5,T=2,p_perception=0.,payoff=PayoffConfig(12,8,0,12,12),drift=DriftConfig(8,0,.55),streak=StreakConfig(.5,0,4),seed=7)
        row=dict(episode_id='OFFLINE_FIXTURE_ONLY',arm='F',model='fixture',parser='final_decision_v2')
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            agents=[Agent(cfg,row,Client(),Tokenizer(),out,16384,2048) for _ in range(2)]
            for ag in agents: ag.base_url='http://127.0.0.1:1/v1'
            result=GameSimulator(cfg).run_episode(agents)
            self.assertTrue(all(r.B_eff==12. and r.true_actions==[4,4] for r in result.logs))
            records=[json.loads(s) for s in (out/'decisions.jsonl').read_text().splitlines()]
            self.assertEqual(len(records),4)
            self.assertTrue(all(r['fallback_used'] and r['repair_calls']==3 for r in records))
            requests=[json.loads(s) for s in (out/'requests.jsonl').read_text().splitlines()]
            self.assertEqual(len(requests),16)
            self.assertTrue(all(r.get('prompt_tokens_server')==3 and r['error'] is None for r in requests))
            self.assertEqual(len({r['request']['seed'] for r in requests}),16)

if __name__=='__main__': unittest.main()
