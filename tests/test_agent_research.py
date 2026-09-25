import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from AI_Agent.agent.llm_agent import LLMAgent
from game_engine.experiments.run_sensitivity import make_turn, scenarios
from game_engine.agents.llm_wrapper import LLMWrapperAgent
from game_engine.env.types import EnvConfig, PayoffConfig, DriftConfig, StreakConfig
from game_engine.env.simulator import GameSimulator
from game_engine.agents.always import AlwaysCooperate
from game_engine.io.jsonl import write_episode

class Client:
    def __init__(self, fail=False):
        self.calls=[]
        self.fail=fail
    def generate(self, system_prompt, user_prompt):
        self.calls.append(json.loads(user_prompt))
        if self.fail:
            raise RuntimeError('network unavailable')
        return json.dumps({'a': 0, 'memory': {'hypothesis': 'tentative', 'evidence': 'x'*500}, 'expectation': 'uncertain'})

class AgentTests(unittest.TestCase):
    def agent(self, directory, client, mode='history_only', **kwargs):
        return LLMAgent(llm_client=client, memory_mode=mode, output_dir=directory,
                        prompt_dir=str(Path(__file__).resolve().parents[1]/'src/AI_Agent/prompts'), **kwargs)

    def test_prediction_ablation_and_single_call(self):
        with tempfile.TemporaryDirectory() as d:
            for enabled in (False, True):
                c = Client()
                a = self.agent(d, c, predict_opponents=enabled)
                a.step(make_turn())
                self.assertEqual(len(c.calls), 1)
                self.assertEqual('expectation' in c.calls[0]['output_schema'], enabled)
                self.assertEqual(a.last_state['predict_opponents'], enabled)

    def test_repair_budget_and_existing_memory_survive_failure(self):
        with tempfile.TemporaryDirectory() as d:
            for budget in (0, 1, 3):
                c = Client(True)
                a = self.agent(d, c, 'model_memory', max_retries=budget)
                a.memory = {'hypothesis': 'previous successful note'}
                self.assertEqual(a.step(make_turn()), 4)
                self.assertEqual(len(c.calls), 1 + budget)
                self.assertEqual(a.last_state['retries'], budget)
                self.assertEqual(len(a.last_state['model_calls']), 1 + budget)
                self.assertEqual(a.memory['hypothesis'], 'previous successful note')
                self.assertTrue(a.last_state['fallback_used'])

    def test_successive_repairs_use_latest_output(self):
        class SequenceClient(Client):
            def generate(self, system_prompt, user_prompt):
                self.calls.append(json.loads(user_prompt))
                return ['bad first', '{"a": 999}', '{"a": "2"}'][len(self.calls)-1]
        with tempfile.TemporaryDirectory() as d:
            c = SequenceClient()
            a = self.agent(d, c)
            self.assertEqual(a.step(make_turn()), 2)
            self.assertEqual(c.calls[1]['previous_invalid_output'], 'bad first')
            self.assertEqual(c.calls[2]['previous_invalid_output'], '{"a": 999}')
            self.assertTrue(a.last_state['valid_model_decision'])

    def test_invalid_repair_budget(self):
        with tempfile.TemporaryDirectory() as d:
            for value in (-1, True, 1.5):
                with self.assertRaises(ValueError):
                    self.agent(d, Client(), max_retries=value)

    def test_memory_conditions_and_reset(self):
        with tempfile.TemporaryDirectory() as d:
            for mode in ('history_only', 'model_memory'):
                c=Client(); a=self.agent(d,c,mode)
                a.step(make_turn()); a.step(make_turn())
                if mode == 'history_only':
                    self.assertNotIn('previous_model_memory', c.calls[-1])
                    self.assertEqual(a.memory,{})
                else:
                    self.assertEqual(c.calls[-1]['previous_model_memory']['hypothesis'],'tentative')
                    self.assertEqual(len(a.memory['evidence']),400)
                    a.reset(); self.assertEqual(a.memory,{})

    def test_noise_and_authoritative_payoff(self):
        with tempfile.TemporaryDirectory() as d:
            c=Client(); a=self.agent(d,c)
            turn=make_turn(p=.4); turn['payoff']['B_effective']=13.7
            a.step(turn)
            self.assertEqual(c.calls[0]['turn']['payoff']['B_effective'],13.7)
            self.assertFalse(c.calls[0]['noise_channel']['execution_noise'])
            self.assertIn('DIFFERENT',c.calls[0]['noise_channel']['observation_rule'])

    def test_failure_is_visible_and_does_not_create_memory(self):
        with tempfile.TemporaryDirectory() as d:
            a=self.agent(d,Client(True),'model_memory')
            self.assertEqual(a.step(make_turn(p=1)),4)
            self.assertFalse(a.last_state['valid_model_decision'])
            self.assertEqual(a.memory,{})

    def test_repair_preserves_memory_condition(self):
        class RepairClient(Client):
            def generate(self, system_prompt, user_prompt):
                result = super().generate(system_prompt, user_prompt)
                return '{"a": true}' if len(self.calls) == 1 else result
        with tempfile.TemporaryDirectory() as d:
            c=RepairClient(); a=self.agent(d,c,'model_memory')
            self.assertEqual(a.step(make_turn()),0)
            self.assertEqual(len(c.calls),2)
            self.assertIn('previous_model_memory',c.calls[1])
            self.assertTrue(a.last_state['valid_model_decision'])
            self.assertEqual(a.memory['hypothesis'],'tentative')

    def test_api_circuit_breaker(self):
        import requests
        from AI_Agent.agent.api_guard import GuardedClient, APIUnavailableError
        class Failing:
            def __init__(self, status=None): self.status=status; self.calls=0
            def generate(self, **kwargs):
                self.calls += 1
                if self.status:
                    response=requests.Response(); response.status_code=self.status
                    raise requests.HTTPError('provider failure',response=response)
                raise requests.ConnectionError('DNS failed')
        for status, expected in ((402,1),(401,1),(403,1),(None,3),(503,3)):
            with tempfile.TemporaryDirectory() as d:
                client=Failing(status)
                agent=self.agent(d,GuardedClient(client))
                with self.assertRaises(APIUnavailableError): agent.step(make_turn())
                self.assertEqual(client.calls,expected)
                trace=json.loads((Path(d)/'agent_traces.jsonl').read_text())
                self.assertTrue(trace['aborted'])
                self.assertNotIn('final_action',trace)

    def test_success_resets_api_failure_counter(self):
        from AI_Agent.agent.api_guard import GuardedClient
        guard=GuardedClient(Client())
        guard.consecutive_failures=2
        guard.generate(system_prompt='',user_prompt='{}')
        self.assertEqual(guard.consecutive_failures,0)

    def test_scenario_semantics(self):
        for dimension, level, turn in scenarios():
            gp=turn['game_parameters']; rows=turn['information_set']['observed_history_last_k']
            self.assertEqual(len(rows),gp['N'])
            mapped=[[gp['action_semantics']['index_to_cooperation'][a] for a in row] for row in rows]
            self.assertEqual(mapped[0],[1,1,0,1])
            self.assertEqual(mapped[1],[0,1,1,1])

    def test_episode_metadata_reports_internal_failure(self):
        cfg=EnvConfig(N=2,M=5,T=2,p_perception=0,payoff=PayoffConfig(12,8,B_min=9,B_max=15),
                      drift=DriftConfig(4,0,.55),streak=StreakConfig(.6,0,4))
        with tempfile.TemporaryDirectory() as d:
            wrapper=LLMWrapperAgent('test',0,cfg,backend='dummy',output_dir=d,
                     prompt_dir=str(Path(__file__).resolve().parents[1]/'src/AI_Agent/prompts'))
            wrapper.llm.llm_client=Client(True)
            result=GameSimulator(cfg).run_episode([wrapper,AlwaysCooperate()])
            self.assertTrue(result.logs[0].agent_meta[0].fallback_used)
            self.assertFalse(result.logs[0].agent_meta[0].parse_ok)
            meta,_=write_episode(d,'test',0,result)
            validity=json.loads(Path(meta).read_text())['model_validity']
            self.assertEqual(validity['fallback_decisions'],2)
            self.assertFalse(validity['valid_for_model_comparison'])

if __name__ == '__main__': unittest.main()
