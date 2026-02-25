from __future__ import annotations

from typing import Any, Optional

from ..graph.runner import GraphRunner
from ..graph.nodes.parse_turn import N0_ParseTurn
from ..graph.nodes.build_features import N1_BuildFeatures
from ..graph.nodes.compute_payoff import N2_ComputePayoff
from ..graph.nodes.compute_noise import N3_ComputeNoise
from ..graph.nodes.opp_model_llm import N4_OpponentModel
from ..graph.nodes.rank_actions import N5_RankActions
from ..graph.nodes.decide_llm import N6_DecisionPolicy
from ..graph.nodes.validate import N7_ValidateDecision
from ..graph.nodes.repair_llm import N8_RepairOrFinalize
from ..graph.nodes.fallback import N9_FallbackDecision
from ..graph.nodes.return_action import N10_ReturnAction

from .llm_client import OllamaClient
from .prompt_loader import load_prompts
from .logger import AgentLogger


class LLMAgent:
    """
    Main agent class.

    Responsibilities:
    - Load prompts
    - Construct graph nodes
    - Execute graph per turn
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
    ) -> None:
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

        # Instantiate all nodes
        self.nodes = {
            "N0_ParseTurn": N0_ParseTurn(),
            "N1_BuildFeatures": N1_BuildFeatures(),
            "N2_ComputePayoff": N2_ComputePayoff(),
            "N3_ComputeNoise": N3_ComputeNoise(),
            "N4_OpponentModel": N4_OpponentModel(),
            "N5_RankActions": N5_RankActions(),
            "N6_DecisionPolicy": N6_DecisionPolicy(),
            "N7_ValidateDecision": N7_ValidateDecision(),
            "N8_RepairOrFinalize": N8_RepairOrFinalize(),
            "N9_FallbackDecision": N9_FallbackDecision(),
            "N10_ReturnAction": N10_ReturnAction(),
        }

        self.runner = GraphRunner(self.nodes)

    def step(self, turn_input: dict) -> int:
        """
        Execute one full graph cycle for a turn.
        Returns chosen action index.
        """
        initial_state = {"raw_input": turn_input}

        context = {
            "llm_client": self.llm_client,
            "prompts": self.prompts,
        }

        final_state = self.runner.run(initial_state, context)
        action = final_state["final_action"]

        # Write outputs (submission + trace)
        self.logger.write_submission(action)
        self.logger.write_trace(final_state)

        return action