from typing import Any, Dict, List
import math

from ..node import Node


class N5_RankActions(Node):
    """
    Strategic action ranking for a repeated game.

    Main change from the old version:
    - Do NOT sort purely by one-step expected utility.
    - Build a reciprocity-aware target cooperation level from:
        * opponent next-action forecast,
        * recent observed opponent behavior,
        * longer-run observed opponent behavior,
        * small inertia from own previous move.
    - Keep one-step EU only as a weak diagnostic / tie-break signal.

    Output:
    - state["candidates"]: list[dict], sorted by descending strategic_score
    - state["strategy_summary"]: compact diagnostics for the decision node
    - state["fallback_action"]: deterministic strategic fallback action
    """

    def __init__(self):
        super().__init__("N5_RankActions")

    @staticmethod
    def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, x))

    @staticmethod
    def _mean_coop_from_rows(
        rows: List[List[int]],
        coop_vals: List[float],
        window: int,
    ) -> float:
        if window <= 0 or not rows:
            return 0.5

        vals: List[float] = []
        for row in rows:
            upto = min(window, len(row))
            for t in range(upto):
                a = row[t]
                if isinstance(a, int) and 0 <= a < len(coop_vals):
                    vals.append(float(coop_vals[a]))

        return float(sum(vals) / len(vals)) if vals else 0.5

    @staticmethod
    def _last_coop(row: List[int], coop_vals: List[float]) -> float:
        if not row:
            return 0.5
        a = row[0]
        if not isinstance(a, int) or not (0 <= a < len(coop_vals)):
            return 0.5
        return float(coop_vals[a])

    def run(self, state: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        # --- Required fields ---
        N = state.get("N")
        M = state.get("M")
        agent_id = state.get("agent_id")
        B_eff = state.get("B_eff")
        C = state.get("C")
        K = state.get("K")
        p = state.get("p")
        round_num = int(state.get("round", 1))

        index_to_coop = state.get("index_to_coop")
        opp_probs = state.get("opponent_action_probs")

        if N is None or M is None or agent_id is None:
            raise RuntimeError("RankActions: missing one of required keys: N, M, agent_id.")
        if B_eff is None or C is None or K is None:
            raise RuntimeError("RankActions: missing payoff parameters: B_eff, C, K.")
        if p is None:
            raise RuntimeError("RankActions: missing noise parameter p.")
        if not isinstance(index_to_coop, list) or len(index_to_coop) != M:
            raise RuntimeError(
                f"RankActions: index_to_coop must be list of length M={M}, "
                f"got {type(index_to_coop)} len={len(index_to_coop) if isinstance(index_to_coop, list) else 'NA'}."
            )
        if not isinstance(opp_probs, dict):
            raise RuntimeError("RankActions: opponent_action_probs missing or not a dict.")

        # --- Noise constants ---
        if p > 0:
            noise_model = state.get("noise_model")
            if not isinstance(noise_model, dict) or "mean_action_coop" not in noise_model:
                raise RuntimeError("RankActions: noise_model['mean_action_coop'] missing but p > 0.")
            mean_action_coop = float(noise_model["mean_action_coop"])
        else:
            mean_action_coop = 0.0  # unused when p == 0

        coop_vals: List[float] = [float(index_to_coop[j]) for j in range(M)]

        # --- Opponent forecast: E[avg_other_true_coop] ---
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
            if any((not isinstance(x, (int, float))) for x in dist):
                raise RuntimeError(
                    f"RankActions: opponent_action_probs[{key}] contains non-numeric values."
                )
            if any(float(x) < -1e-12 for x in dist):
                raise RuntimeError(
                    f"RankActions: opponent_action_probs[{key}] contains negative probabilities."
                )

            s = float(sum(dist))
            if (not math.isfinite(s)) or abs(s - 1.0) > 1e-2:
                raise RuntimeError(
                    f"RankActions: opponent_action_probs[{key}] probabilities must sum to 1 (±1e-2). Got {s}."
                )

            expected_i_nominal = sum(float(dist[j]) * coop_vals[j] for j in range(M))
            if p > 0:
                expected_i_true = (1.0 - p) * expected_i_nominal + p * mean_action_coop
            else:
                expected_i_true = expected_i_nominal

            expected_other_true_sum += expected_i_true
            opp_count += 1

        if opp_count != (N - 1):
            raise RuntimeError(
                f"RankActions: opponent count mismatch. Expected {N - 1}, got {opp_count}."
            )

        E_other_true_coop = expected_other_true_sum / opp_count

        # --- Observed history features (most recent first) ---
        raw_history = state.get("info", {}).get("observed_history_last_k", [])
        if not isinstance(raw_history, list) or len(raw_history) != N:
            history: List[List[int]] = [[] for _ in range(N)]
        else:
            history = [row if isinstance(row, list) else [] for row in raw_history]

        self_hist = history[agent_id] if 0 <= agent_id < len(history) else []
        other_rows = [history[i] for i in range(N) if i != agent_id]

        k = len(self_hist)
        recent_window = min(3, k) if k > 0 else 0
        long_window = min(8, k) if k > 0 else 0

        recent_other_coop = (
            self._mean_coop_from_rows(other_rows, coop_vals, recent_window)
            if recent_window > 0 else 0.5
        )
        long_other_coop = (
            self._mean_coop_from_rows(other_rows, coop_vals, long_window)
            if long_window > 0 else recent_other_coop
        )
        last_other_coop = self._mean_coop_from_rows(other_rows, coop_vals, 1) if other_rows else 0.5

        recent_self_coop = (
            self._mean_coop_from_rows([self_hist], coop_vals, recent_window)
            if recent_window > 0 else 0.5
        )
        last_self_action = self_hist[0] if self_hist else None
        last_self_coop = self._last_coop(self_hist, coop_vals) if self_hist else None

        # --- Strategic summary ---
        # Opponent cooperation belief: blend model-based forecast with observed recent behavior.
        belief_other_coop = self._clamp(
            0.50 * E_other_true_coop
            + 0.35 * recent_other_coop
            + 0.15 * long_other_coop
        )

        # If opponents recently got much harsher than their longer-run average, retaliate more.
        downward_shift = max(0.0, long_other_coop - recent_other_coop)
        retaliation_pressure = self._clamp(
            max(0.0, (0.45 - recent_other_coop) / 0.45) + min(0.30, downward_shift)
        )

        # If opponents are cooperating well, gently encourage forgiveness / re-cooperation.
        forgiveness_pressure = self._clamp(
            max(0.0, (recent_other_coop - 0.60) / 0.40)
        )

        if round_num <= 1:
            # Cooperative prior on the first move.
            target_coop = max(0.75, belief_other_coop)
        else:
            target_coop = belief_other_coop
            target_coop = target_coop - 0.30 * retaliation_pressure + 0.10 * forgiveness_pressure
            if last_self_coop is not None:
                # Small inertia so we do not oscillate wildly from round to round.
                target_coop = 0.80 * target_coop + 0.20 * last_self_coop
            target_coop = self._clamp(target_coop)

        # --- Myopic EU remains only a weak feature / diagnostic ---
        raw_eus: List[float] = []
        own_true_coops: List[float] = []

        for a in range(M):
            own_nominal = coop_vals[a]
            if p > 0:
                own_true = (1.0 - p) * own_nominal + p * mean_action_coop
            else:
                own_true = own_nominal

            EU = float(B_eff) * E_other_true_coop - float(C) * own_true + float(K)
            own_true_coops.append(float(own_true))
            raw_eus.append(float(EU))

        eu_min = min(raw_eus)
        eu_max = max(raw_eus)
        eu_span = eu_max - eu_min

        candidates: List[Dict[str, Any]] = []

        for a in range(M):
            own_true = own_true_coops[a]
            EU = raw_eus[a]
            normalized_eu = 0.5 if eu_span < 1e-12 else (EU - eu_min) / eu_span

            reciprocity_score = 1.0 - abs(own_true - target_coop)
            stability_score = (
                0.5 if last_self_coop is None else (1.0 - abs(own_true - last_self_coop))
            )

            # Gentle bias toward cooperative matching when the band is close,
            # and toward punishment only when opponents are consistently harsh.
            phase_bias = own_true if target_coop >= 0.5 else (1.0 - own_true)

            strategic_score = (
                0.60 * reciprocity_score
                + 0.15 * stability_score
                + 0.15 * normalized_eu
                + 0.10 * phase_bias
            )

            if round_num <= 1:
                strategic_score += 0.10 * own_true  # cooperative first move prior

            if round_num > 1 and recent_other_coop < 0.25:
                strategic_score += 0.08 * (1.0 - own_true)  # harsher when opponents are very exploitative

            strategic_score = float(strategic_score)

            candidates.append(
                {
                    "action": int(a),
                    "strategic_score": strategic_score,
                    "distance_to_target": float(abs(own_true - target_coop)),
                    "own_true_coop": float(own_true),
                    "target_coop": float(target_coop),
                    "EU": float(EU),  # kept for logging / analysis only
                    "E_other_true_coop": float(E_other_true_coop),
                    "reciprocity_score": float(reciprocity_score),
                    "stability_score": float(stability_score),
                    "normalized_eu": float(normalized_eu),
                }
            )

        # Strategic ranking first. On near-ties, prefer closer-to-target and then more cooperative action.
        candidates.sort(
            key=lambda x: (
                -x["strategic_score"],
                x["distance_to_target"],
                x["action"],  # lower index = more cooperative
            )
        )

        state["candidates"] = candidates
        state["candidate_ranking_mode"] = "strategic_reciprocity"
        state["strategy_summary"] = {
            "round": round_num,
            "belief_other_coop": float(belief_other_coop),
            "expected_other_true_coop": float(E_other_true_coop),
            "recent_other_coop": float(recent_other_coop),
            "long_other_coop": float(long_other_coop),
            "last_other_coop": float(last_other_coop),
            "recent_self_coop": float(recent_self_coop),
            "last_self_action": last_self_action,
            "last_self_coop": None if last_self_coop is None else float(last_self_coop),
            "retaliation_pressure": float(retaliation_pressure),
            "forgiveness_pressure": float(forgiveness_pressure),
            "target_coop": float(target_coop),
        }
        state["fallback_action"] = int(candidates[0]["action"])
        return state