from __future__ import annotations

from typing import Any, Optional

from .pipeline import run_turn

from .api_guard import APIUnavailableError
from .llm_client import OllamaClient
from .prompt_loader import load_prompts
from .logger import AgentLogger


class LLMAgent:
    """
    Main agent class.

    Responsibilities:
    - Load prompts
    - Execute the decision pipeline per turn
    - Log outputs

    LLM client is injected (so you can do model-vs-model and swap backends).
    Backward-compatible: if llm_client is not provided, defaults to OllamaClient(model_name).
    """

    def __init__(
        self,
        *,
        llm_client: Optional[Any] = None,
        model_name: str = "llama3.1:8b",
        ollama_host: str = "http://localhost:11434",
        prompt_dir: str = "AI_Agent/prompts",
        logger: Optional[AgentLogger] = None,
        output_dir: Optional[str] = None,
        memory_mode: str = "history_only",
        predict_opponents: bool = False,
        max_retries: int = 3,
    ) -> None:
        if memory_mode not in ("history_only", "model_memory"):
            raise ValueError("Unknown memory condition")
        if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
            raise ValueError("max_retries must be a nonnegative integer")
        self.max_retries = max_retries
        self.predict_opponents = predict_opponents
        self.memory_mode = memory_mode
        self.memory = {}
        self.last_state = {}
        # Client injection (preferred)
        if llm_client is not None:
            self.llm_client = llm_client
        else:
            # Backward-compatible default (no longer "hardcoded" in the sense of "only Ollama forever")
            self.llm_client = OllamaClient(model_name=model_name, host=ollama_host)

        self.prompts = load_prompts(prompt_dir)

        # Logging: allow wrapper/runner to isolate directories
        if logger is not None:
            self.logger = logger
        else:
            self.logger = AgentLogger(output_dir=output_dir or "AI_Agent/outputs")

    def step(self, turn_input: dict) -> int:
        """
        Execute one decision pipeline for a turn.
        Returns chosen action index.
        """
        initial_state = {"raw_input": turn_input, "memory": dict(self.memory)}

        context = {
            "llm_client": self.llm_client,
            "prompts": self.prompts,
            "memory_mode": self.memory_mode,
            "predict_opponents": self.predict_opponents,
        }

        try:
            final_state = run_turn(initial_state, context, max_retries=self.max_retries)
        except APIUnavailableError as exc:
            initial_state.update(aborted=True, valid_model_decision=False,
                                 memory_mode=self.memory_mode, api_error=str(exc))
            self.last_state = initial_state
            self.logger.write_trace(initial_state)
            raise
        action = final_state["final_action"]
        fallback = final_state["fallback_used"]
        final_state["fallback_used"] = fallback
        final_state["memory_mode"] = self.memory_mode
        final_state["predict_opponents"] = self.predict_opponents
        final_state["valid_model_decision"] = not fallback
        self.last_state = final_state
        if self.memory_mode == "model_memory" and not fallback:
            candidate = final_state["decision"].get("memory", {})
            if isinstance(candidate, dict):
                self.memory = {key: value[:400] for key, value in candidate.items()
                               if key in ("hypothesis", "evidence", "uncertainty", "reconsider_if")
                               and isinstance(value, str)}

        # Write outputs (submission + trace)
        self.logger.write_submission(action)
        self.logger.write_trace(final_state)

        return action

    def reset(self, seed=None):
        self.memory = {}
        self.last_state = {}
