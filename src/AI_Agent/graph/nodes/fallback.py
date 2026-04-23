from ..node import Node


class N9_FallbackDecision(Node):
    """
    Deterministic fallback if the LLM fails repeatedly.

    This fallback does NOT use opponent labels, opponent forecasts, or candidate rankings.
    It uses only raw payoff parameters plus the recent observed mean cooperation of others.
    """

    def __init__(self):
        super().__init__("N9_FallbackDecision")

    def run(self, state, context):
        M = int(state["M"])
        B_eff = float(state.get("B_eff", 0.0))
        C = float(state.get("C", 0.0))
        K = float(state.get("K", 0.0))
        p = float(state.get("p", 0.0))

        coop_vals = [float(x) for x in state.get("index_to_coop", list(range(M)))]
        features = state.get("features", {})
        mean_other_obs = float(features.get("mean_other_coop_recent", 0.5))

        mean_action_coop = None
        if p > 0:
            noise_model = state.get("noise_model", {})
            mean_action_coop = float(
                noise_model.get("mean_action_coop", sum(coop_vals) / len(coop_vals))
            )

        scored = []
        for a in range(M):
            own_nominal = coop_vals[a]
            if p > 0 and mean_action_coop is not None:
                own_true = (1.0 - p) * own_nominal + p * mean_action_coop
            else:
                own_true = own_nominal

            score = B_eff * mean_other_obs - C * own_true + K
            scored.append((float(score), int(a)))

        best_score = max(score for score, _ in scored)
        near_best = [a for score, a in scored if score >= best_score - 1e-9]
        chosen = min(near_best)  # cooperative tie-break

        state["decision"] = {
            "a": int(chosen),
            "reason": (
                "Fallback: repeated JSON failure; choosing the action with the "
                "best immediate payoff under recent observed cooperation, with "
                "a cooperative tie-break."
            ),
            "confidence": 0.50,
        }
        state["is_valid"] = True
        return state