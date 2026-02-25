from ..node import Node
import numpy as np


class N1_BuildFeatures(Node):
    """
    Build structured features from observed history.

    Computes:
    - Recent cooperation per player
    - Mean cooperation excluding self
    - Most recent observed actions
    """

    def __init__(self):
        super().__init__("N1_BuildFeatures")

    def run(self, state, context):

        history = state["info"]["observed_history_last_k"]
        index_to_coop = state["index_to_coop"]
        agent_id = state["agent_id"]

        N = state["N"]
        k = len(history[0])

        coop_matrix = np.zeros((N, k))

        for i in range(N):
            for t in range(k):
                coop_matrix[i, t] = index_to_coop[history[i][t]]

        # Most recent cooperation (t=0 is most recent)
        recent_coop = coop_matrix[:, 0]

        # Mean observed cooperation excluding self
        others = [i for i in range(N) if i != agent_id]
        mean_other_coop_recent = np.mean(recent_coop[others])

        state["features"]["recent_coop"] = recent_coop.tolist()
        state["features"]["mean_other_coop_recent"] = float(mean_other_coop_recent)

        return state