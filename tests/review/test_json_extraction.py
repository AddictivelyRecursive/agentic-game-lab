import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code" / "src"))
from AI_Agent.agent.pipeline import extract_decision_json

class ExtractionTests(unittest.TestCase):
    def test_reasoning_braces_and_fenced_final(self):
        result = extract_decision_json('<think>{not json} {"a": 0}</think>\n```json\n{"a":4,"reason":"brace } inside string"}\n```')
        self.assertEqual(result["a"], 4)
    def test_plain_and_unmarked_reasoning(self):
        self.assertEqual(extract_decision_json('Analysis {invalid}. Final: {"a":2}')["a"], 2)
        self.assertEqual(extract_decision_json('{"a":0}')["a"], 0)
    def test_reject_ambiguous_truncated_and_unfinished_reasoning(self):
        for text in ('{"a":0} {"a":4}', '{"a":4,"reason":"cut', '<think>{"a":4}', 'No answer'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                extract_decision_json(text)
