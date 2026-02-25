from ..node import Node


class N3_ComputeNoise(Node):
    """
    Build noise model.

    Since p applies to both execution and observation,
    we approximate true cooperation expectation as:

    E_true = (1 - p) * coop + p * mean_action_coop

    where mean_action_coop is average cooperation over action space.
    """

    def __init__(self):
        super().__init__("N3_ComputeNoise")

    def run(self, state, context):

        p = state["p"]
        index_to_coop = state["index_to_coop"]

        mean_action_coop = sum(index_to_coop) / len(index_to_coop)

        state["noise_model"] = {
            "p": p,
            "mean_action_coop": mean_action_coop
        }

        return state