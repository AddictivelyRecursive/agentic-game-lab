from ..node import Node
import numpy as np


class N1_BuildFeatures(Node):
    """
    Build structured features from observed history.

    Computes:
    - Recent cooperation per player
    - Mean cooperation excluding self
    - Most recent observed actions
    - Robust empty-history defaults for t=1
    """

    def __init__(self):
        super().__init__("N1_BuildFeatures")

    def run(self, state, context):
        history = state["info"]["observed_history_last_k"]
        index_to_coop = state["index_to_coop"]
        agent_id = state["agent_id"]
        N = state["N"]

        if not isinstance(history, list) or len(history) != N:
            history = [[] for _ in range(N)]
        else:
            history = [row if isinstance(row, list) else [] for row in history]

        k = max((len(row) for row in history), default=0)

        if k > 0:
            coop_matrix = np.full((N, k), 0.5, dtype=float)
            for i in range(N):
                for t, a in enumerate(history[i]):
                    if isinstance(a, int) and 0 <= a < len(index_to_coop):
                        coop_matrix[i, t] = float(index_to_coop[a])
            recent_coop = coop_matrix[:, 0]
        else:
            coop_matrix = np.zeros((N, 0), dtype=float)
            recent_coop = np.full((N,), 0.5, dtype=float)

        most_recent_actions = [
            row[0] if row else None
            for row in history
        ]

        others = [i for i in range(N) if i != agent_id]
        mean_other_coop_recent = (
            float(np.mean(recent_coop[others])) if others else 0.5
        )

        state["features"] = {
            "recent_coop": recent_coop.tolist(),
            "mean_other_coop_recent": mean_other_coop_recent,
            "most_recent_actions": most_recent_actions,
            "history_length": int(k),
            "has_history": bool(k > 0),
        }
        return state