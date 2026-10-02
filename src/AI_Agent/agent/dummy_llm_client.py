import json
import random
from typing import Optional


class DummyLLMClient:
    """
    A pseudo-LLM used for testing the agent end-to-end without Ollama.

    It can simulate:
    - correct JSON
    - malformed output (to trigger repair)
    - wrong action range (to trigger repair/fallback)

    Usage:
        llm = DummyLLMClient(mode="mostly_valid", seed=42)
        text = llm.generate(system_prompt="...", user_prompt="...")
    """

    def __init__(
        self,
        mode: str = "always_valid",
        seed: Optional[int] = 0,
        invalid_rate: float = 0.3,
        force_invalid_first_calls: int = 0,
    ):
        """
        Parameters
        ----------
        mode : str
            One of:
            - "always_valid": always returns valid decision JSON
            - "mostly_valid": sometimes returns invalid output (invalid_rate)
            - "always_invalid": always returns invalid output (useful to test fallback)
        seed : Optional[int]
            Seed for deterministic randomness.
        invalid_rate : float
            For "mostly_valid", probability of emitting invalid output per call.
        force_invalid_first_calls : int
            Force the first K decision or repair calls to be invalid.
            Helps test repair flow deterministically.
        """
        self.mode = mode
        self.invalid_rate = invalid_rate
        self.rng = random.Random(seed)
        self.force_invalid_first_calls = force_invalid_first_calls
        self._calls = 0

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return self._respond_decision(user_prompt)

    def _should_be_invalid(self) -> bool:
        if self.mode == "always_valid":
            return False
        if self.mode == "always_invalid":
            return True
        if self.mode == "mostly_valid":
            if self._calls <= self.force_invalid_first_calls:
                return True
            return self.rng.random() < self.invalid_rate
        return False

    def _extract_M(self, text: str) -> int:
        return int(json.loads(text)["turn"]["game_parameters"]["M"])

    def _respond_decision(self, user_prompt: str) -> str:
        self._calls += 1
        M = self._extract_M(user_prompt)

        # decide whether to emit invalid
        if self._should_be_invalid():
            # cycle through a few failure modes
            mode = self.rng.choice(["non_json", "out_of_range", "wrong_key", "string_a"])
            if mode == "non_json":
                return "I choose action 2 because it seems best."  # no JSON
            if mode == "out_of_range":
                return json.dumps({"a": M + 5, "reason": "oops", "confidence": 0.1})
            if mode == "wrong_key":
                return json.dumps({"action": 2, "reason": "wrong key", "confidence": 0.2})
            if mode == "string_a":
                return json.dumps({"a": "2", "reason": "string but parseable", "confidence": 0.4})

        # valid response
        a = 0 if M <= 1 else 2 % M
        obj = {"a": a, "reason": "Dummy policy: choosing a stable mid action.", "confidence": 0.6}
        return json.dumps(obj)
