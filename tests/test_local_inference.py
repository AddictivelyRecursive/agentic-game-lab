"""Offline inference-boundary tests; never load weights or contact a provider."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from AI_Agent.agent.local_openai_client import LocalOpenAIClient
from AI_Agent.agent.api_guard import APIUnavailableError
from AI_Agent.model_registry import MODEL_REGISTRY, resolve_model
from game_engine.agents.llm_wrapper import LLMWrapperAgent
from game_engine.experiments.run_local_episode import load_config, run
from game_engine.env.types import EnvConfig


class LocalInferenceTests(unittest.TestCase):
    def test_chat_payload_and_explicit_generation_without_keys(self):
        with patch.dict(os.environ, {}, clear=True):
            client = LocalOpenAIClient("qwen3-8b", "http://localhost:18765/v1/",
                generation_config={"temperature": .3, "top_p": .9, "max_tokens": 1000, "seed": 42}, timeout_s=75)
        response = Mock(status_code=200)
        response.json.return_value = {"choices": [{"message": {"content": '{"a": 2}'}}]}
        with patch.object(client.session, "post", return_value=response) as post:
            self.assertEqual(client.generate("system unchanged", "history unchanged"), '{"a": 2}')
        post.assert_called_once_with("http://localhost:18765/v1/chat/completions", json={
            "model": "qwen3-8b", "messages": [
                {"role": "system", "content": "system unchanged"},
                {"role": "user", "content": "history unchanged"}],
            "temperature": .3, "top_p": .9, "max_tokens": 1000, "seed": 42,
        }, timeout=75, allow_redirects=False)
        self.assertFalse(client.session.trust_env)
        self.assertNotIn("Authorization", client.session.headers)

    def test_external_urls_and_redirects_are_rejected(self):
        for url in ("https://api.example.org/v1", "http://example.org/v1", "http://localhost.evil/v1",
                    "http://user:secret@localhost/v1", "http://localhost/v1?forward=1"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                LocalOpenAIClient("test", url)
        client = LocalOpenAIClient("test", "http://127.0.0.1:8001/v1")
        with patch.object(client.session, "post", return_value=Mock(status_code=302)):
            with self.assertRaises(requests.HTTPError):
                client.generate("s", "u")

    def test_http_failure_propagates_without_retry_or_provider_switch(self):
        client = LocalOpenAIClient("test", "http://localhost:8001/v1")
        with patch.object(client.session, "post", side_effect=requests.Timeout("timeout")) as post:
            with self.assertRaises(requests.Timeout):
                client.generate("s", "u")
            self.assertEqual(post.call_count, 1)

    def test_local_wrapper_uses_env_endpoint_and_model_without_key(self):
        _, environment = load_config(ROOT / "experiments/local_smoke.json")
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {
            "LLM_BASE_URL": "http://localhost:18888/v1", "LLM_MODEL": "qwen3-8b"}, clear=True):
            agent = LLMWrapperAgent("test", 0, EnvConfig(**environment), backend="local",
                output_dir=d, prompt_dir=str(ROOT / "src/AI_Agent/prompts"))
            self.assertIsInstance(agent.llm.llm_client, LocalOpenAIClient)
            self.assertEqual(agent.llm.llm_client.model_name, "qwen3-8b")
            self.assertEqual(agent.llm.llm_client.base_url, "http://localhost:18888/v1")

    def test_invalid_generation_is_rejected(self):
        for config in ({"seed": True}, {"max_tokens": 0}, {"temperature": float("nan")},
                       {"top_p": 0}, {"messages": []}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                LocalOpenAIClient("test", "http://localhost:8001/v1", generation_config=config)

    def test_registry_is_complete_and_does_not_probe_cache(self):
        self.assertEqual(len(MODEL_REGISTRY), 25)
        self.assertEqual(len({m['repo'] for m in MODEL_REGISTRY.values()}), 25)
        with patch("pathlib.Path.exists", side_effect=AssertionError("No cache checks during resolution")):
            self.assertEqual(resolve_model("qwen3-8b")["repo"], "Qwen/Qwen3-8B")
        self.assertEqual(resolve_model("qwen2.5-72b-instruct")["tp_size"], 2)
        with self.assertRaises(ValueError):
            resolve_model("typo")

    def test_mocked_smoke_preserves_history_repair_and_result_metadata(self):
        spec, environment = load_config(ROOT / "experiments/local_smoke.json")
        self.assertEqual((environment['N'], environment['T'], spec['seeds']), (2, 5, [101]))
        calls = []
        def generate(client, system_prompt, user_prompt, temperature=None):
            calls.append(json.loads(user_prompt))
            return '{"a": 999}' if len(calls) == 1 else '{"a": 2, "reason": "mock"}'
        with tempfile.TemporaryDirectory() as d, patch.object(LocalOpenAIClient, "generate", generate):
            output = Path(d) / "smoke"
            run(spec, environment, model="qwen3-8b", base_url="http://localhost:18888/v1", output=output)
            meta = json.loads((output / 'seed_101/episode_101_meta.json').read_text())
            self.assertEqual(meta['num_rounds'], 5)
            self.assertEqual(meta['model_validity']['llm_decisions'], 10)
            self.assertTrue(meta['model_validity']['valid_for_model_comparison'])
            traces = [json.loads(s) for s in (output / 'seed_101/agents/p0/agent_traces.jsonl').read_text().splitlines()]
            self.assertEqual(traces[0]['retries'], 1)
            self.assertEqual(traces[0]['eu_ranking'], calls[0]['expected_utility_ranking'])
            self.assertEqual(calls[1]['expected_utility_ranking'], calls[0]['expected_utility_ranking'])
            last_history = calls[-1]['turn']['information_set']['observed_history_last_k']
            self.assertEqual(last_history, [[2] * 4, [2] * 4])
            self.assertEqual(len(calls), 11)
            # Reusing output paths must never overwrite a completed run.
            with self.assertRaises(FileExistsError):
                run(spec, environment, model="test", base_url="http://localhost:18888/v1", output=output)

    def test_abort_keeps_completed_episode_and_partial_agent_trace(self):
        spec, environment = load_config(ROOT / "experiments/local_smoke.json")
        spec['seeds'] = [101, 102]
        responses = ['{"a": 2}'] * 10 + [requests.ConnectionError("offline")] * 3
        with tempfile.TemporaryDirectory() as d, patch.object(LocalOpenAIClient, "generate", side_effect=responses):
            output = Path(d) / "failed"
            with self.assertRaises(APIUnavailableError):
                run(spec, environment, model="test", base_url="http://localhost:18888/v1", output=output)
            self.assertTrue((output / 'seed_101/episode_101_meta.json').exists())
            self.assertTrue((output / 'error.json').exists())
            trace = json.loads((output / 'seed_102/agents/p0/agent_traces.jsonl').read_text())
            self.assertTrue(trace['aborted'])

    def test_fallback_is_saved_but_run_fails(self):
        spec, environment = load_config(ROOT / "experiments/local_smoke.json")
        with tempfile.TemporaryDirectory() as d, patch.object(LocalOpenAIClient, "generate", return_value='invalid'):
            output = Path(d) / "fallback"
            with self.assertRaisesRegex(RuntimeError, "fallback"):
                run(spec, environment, model="test", base_url="http://localhost:18888/v1", output=output)
            meta = json.loads((output / 'seed_101/episode_101_meta.json').read_text())
            self.assertFalse(meta['model_validity']['valid_for_model_comparison'])


if __name__ == '__main__':
    unittest.main()
