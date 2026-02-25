from ..node import Node

class N0_ParseTurn(Node):
    """
    Parse the incoming JSON turn input and initialize internal state structure.

    This node:
    - Copies game parameters into structured state fields
    - Extracts payoff parameters
    - Extracts information_set
    - Initializes placeholders for later nodes
    """

    def __init__(self):
        super().__init__("N0_ParseTurn")

    def run(self, state, context):

        turn = state["raw_input"]

        # Basic metadata
        state["round"] = turn["round"]
        state["agent_id"] = turn["agent_id"]

        # Game parameters
        state["game_parameters"] = turn["game_parameters"]
        state["N"] = turn["game_parameters"]["N"]
        state["M"] = turn["game_parameters"]["M"]
        state["p"] = turn["game_parameters"]["perception_noise_p"]
        state["index_to_coop"] = turn["game_parameters"]["action_semantics"]["index_to_cooperation"]

        # Payoff structure
        state["payoff"] = turn["payoff"]

        # Streak + drift rules
        state["streak_rule"] = turn["streak_rule"]
        state["streak_effect"] = turn["streak_effect_on_payoff"]
        state["drift_rule"] = turn["drift_rule"]

        # Information set
        state["info"] = turn["information_set"]

        # Placeholders
        state["features"] = {}
        state["noise_model"] = {}
        state["candidates"] = []
        state["decision"] = None
        state["is_valid"] = False

        return state