from ..node import Node
from typing import List, Dict, Any
import math


class N5_RankActions(Node):
    """
    Deterministic action ranking (strict research mode).

    Computes expected utility (EU) for each action a in [0, M-1] using:
        u_i = B_eff * avg_other_coop(true) - C * own_coop(true) + K

    Inputs required in `state` (assumed correct; fail-fast if missing):
    - N, M, agent_id
    - B_eff, C, K
    - p
    - index_to_coop : list length M mapping action index -> cooperation in [0,1]
    - noise_model["mean_action_coop"] if p > 0
    - opponent_action_probs : dict mapping opponent_id (as str) -> list[float] length M
      (This should be produced by N4_OpponentModel; N4 is responsible for fallback if LLM fails.)

    Noise model (both execution + observation noise), approximate true cooperation expectation as:
        E_true(coop) = (1 - p) * E_nominal(coop) + p * mean_action_coop
    where mean_action_coop is the average cooperation over the action space.

    Outputs:
    - state["candidates"]: list of dicts sorted by descending EU:
        [{"action": int, "EU": float, "own_true_coop": float, "E_other_true_coop": float}, ...]
    """

    def __init__(self):
        super().__init__("N5_RankActions")

    def run(self, state: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        # --- Required fields ---
        N = state.get("N")
        M = state.get("M")
        agent_id = state.get("agent_id")
        B_eff = state.get("B_eff")
        C = state.get("C")
        K = state.get("K")
        p = state.get("p")
        index_to_coop = state.get("index_to_coop")
        opp_probs = state.get("opponent_action_probs")

        if N is None or M is None or agent_id is None:
            raise RuntimeError("RankActions: Missing one of required keys: N, M, agent_id.")
        if B_eff is None or C is None or K is None:
            raise RuntimeError("RankActions: Missing payoff parameters: B_eff, C, K.")
        if p is None:
            raise RuntimeError("RankActions: Missing noise parameter p.")
        if not isinstance(index_to_coop, list) or len(index_to_coop) != M:
            raise RuntimeError(
                f"RankActions: index_to_coop must be list of length M={M}, got {type(index_to_coop)} len={len(index_to_coop) if isinstance(index_to_coop, list) else 'NA'}."
            )
        if not isinstance(opp_probs, dict):
            raise RuntimeError("RankActions: opponent_action_probs missing or not a dict. N4 must set it.")

        # --- Noise constants ---
        if p > 0:
            noise_model = state.get("noise_model")
            if not isinstance(noise_model, dict) or "mean_action_coop" not in noise_model:
                raise RuntimeError("RankActions: noise_model['mean_action_coop'] missing but p > 0.")
            mean_action_coop = float(noise_model["mean_action_coop"])
        else:
            mean_action_coop = 0.0  # unused

        coop_vals: List[float] = [float(index_to_coop[j]) for j in range(M)]

        # --- Validate opponent distributions and compute E[avg_other_true_coop] ---
        expected_other_true_sum = 0.0
        opp_count = 0

        for i in range(N):
            if i == agent_id:
                continue

            key = str(i)
            if key not in opp_probs:
                raise RuntimeError(
                    f"RankActions: opponent_action_probs missing entry for opponent player_id={i}."
                )

            dist = opp_probs[key]
            if not isinstance(dist, list) or len(dist) != M:
                raise RuntimeError(
                    f"RankActions: opponent_action_probs[{key}] must be list length M={M}."
                )

            # Basic validity checks: non-negative and sums ~ 1
            if any((not isinstance(x, (int, float))) for x in dist):
                raise RuntimeError(f"RankActions: opponent_action_probs[{key}] contains non-numeric values.")
            if any(x < -1e-12 for x in dist):
                raise RuntimeError(f"RankActions: opponent_action_probs[{key}] contains negative probabilities.")

            s = float(sum(dist))
            if not math.isfinite(s) or abs(s - 1.0) > 1e-2:
                # In strict mode, treat this as an error (N4 should normalize/repair/fallback)
                raise RuntimeError(
                    f"RankActions: opponent_action_probs[{key}] probabilities must sum to 1 (±1e-2). Got {s}."
                )

            expected_i_nominal = sum(float(dist[j]) * coop_vals[j] for j in range(M))

            # Apply noise to approximate true cooperation expectation
            if p > 0:
                expected_i_true = (1.0 - p) * expected_i_nominal + p * mean_action_coop
            else:
                expected_i_true = expected_i_nominal

            expected_other_true_sum += expected_i_true
            opp_count += 1

        if opp_count != (N - 1):
            raise RuntimeError(f"RankActions: opponent count mismatch. Expected {N-1}, got {opp_count}.")

        E_other_true_coop = expected_other_true_sum / opp_count

        # --- Compute EU per own action ---
        candidates = []
        for a in range(M):
            own_nominal = coop_vals[a]
            if p > 0:
                own_true = (1.0 - p) * own_nominal + p * mean_action_coop
            else:
                own_true = own_nominal

            EU = float(B_eff) * E_other_true_coop - float(C) * own_true + float(K)

            candidates.append({
                "action": int(a),
                "EU": float(EU),
                "own_true_coop": float(own_true),
                "E_other_true_coop": float(E_other_true_coop),
            })

        # Sort descending by EU, stable tie-break by lower action index (more cooperation)
        candidates.sort(key=lambda x: (-x["EU"], x["action"]))

        state["candidates"] = candidates
        return state