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
    - Initialize LLM client
    - Load prompts
    - Construct graph nodes
    - Execute graph per turn
    - Log outputs
    """

    def __init__(self,
                 model_name="llama3.1:8b",
                 prompt_dir="AI_Agent/prompts"):

        self.llm_client = OllamaClient(model_name=model_name)
        self.prompts = load_prompts(prompt_dir)
        self.logger = AgentLogger()

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
            "prompts": self.prompts
        }

        final_state = self.runner.run(initial_state, context)

        action = final_state["final_action"]

        # Write outputs
        self.logger.write_submission(action)
        self.logger.write_trace(final_state)

        return action