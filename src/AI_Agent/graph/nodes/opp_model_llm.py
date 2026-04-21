from __future__ import annotations

import math
from typing import Any, Dict, List

from ..node import Node


class N4_OpponentModel(Node):
    """
    Deterministic opponent-style modeling.

    Replaces the old LLM-based N4 with a compact, reproducible summary of each
    opponent's observed behavior and a deterministic next-action distribution.

    Outputs:
    - state["opponent_action_probs"]: dict[str, list[float]]
    - state["opponent_style_summary"]: dict[str, dict]
    """

    def __init__(self):
        super().__init__("N4_OpponentModel")

    @staticmethod
    def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, float(x)))

    @staticmethod
    def _normalize(dist: List[float]) -> List[float]:
        if not dist:
            return []
        clipped = [max(0.0, float(x)) for x in dist]
        s = sum(clipped)
        if s <= 1e-12:
            return [1.0 / len(clipped)] * len(clipped)
        return [x / s for x in clipped]

    @staticmethod
    def _mean_coop(row: List[int], coop_vals: List[float], window: int | None = None) -> float:
        if not row:
            return 0.5
        upto = len(row) if window is None else min(window, len(row))
        vals: List[float] = []
        for t in range(upto):
            a = row[t]
            if isinstance(a, int) and 0 <= a < len(coop_vals):
                vals.append(float(coop_vals[a]))
        return float(sum(vals) / len(vals)) if vals else 0.5

    @staticmethod
    def _switch_rate(row: List[int]) -> float:
        if len(row) <= 1:
            return 0.0
        switches = 0
        pairs = 0
        for t in range(len(row) - 1):
            a0 = row[t]
            a1 = row[t + 1]
            if isinstance(a0, int) and isinstance(a1, int):
                pairs += 1
                if a0 != a1:
                    switches += 1
        return float(switches / pairs) if pairs else 0.0

    @staticmethod
    def _soft_dist_from_target(target_coop: float, coop_vals: List[float], sharpness: float = 8.0) -> List[float]:
        scores = [math.exp(-sharpness * abs(float(c) - float(target_coop))) for c in coop_vals]
        return N4_OpponentModel._normalize(scores)

    @staticmethod
    def _weighted_average(parts: List[tuple[List[float], float]]) -> List[float]:
        usable = [(d, float(w)) for d, w in parts if d and float(w) > 0.0]
        if not usable:
            return []
        M = len(usable[0][0])
        out = [0.0] * M
        total_w = 0.0
        for dist, w in usable:
            total_w += w
            for j in range(M):
                out[j] += float(dist[j]) * w
        if total_w <= 1e-12:
            return [1.0 / M] * M
        out = [x / total_w for x in out]
        return N4_OpponentModel._normalize(out)

    @staticmethod
    def _argmax(dist: List[float]) -> int:
        best_idx = 0
        best_val = dist[0]
        for i, x in enumerate(dist):
            if x > best_val:
                best_idx = i
                best_val = x
        return int(best_idx)

    def run(self, state: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        agent_id = int(state["agent_id"])
        N = int(state["N"])
        M = int(state["M"])

        index_to_coop = state["index_to_coop"]
        coop_vals = [float(x) for x in index_to_coop]

        info = state.get("info", {})
        action_freq = info.get("action_freq_NxM", [])
        raw_history = info.get("observed_history_last_k", [])

        if not isinstance(raw_history, list) or len(raw_history) != N:
            history: List[List[int]] = [[] for _ in range(N)]
        else:
            history = [row if isinstance(row, list) else [] for row in raw_history]

        uniform = [1.0 / M] * M
        opponent_action_probs: Dict[str, List[float]] = {}
        opponent_style_summary: Dict[str, Dict[str, Any]] = {}

        for i in range(N):
            if i == agent_id:
                continue

            row = history[i]

            if (
                isinstance(action_freq, list)
                and i < len(action_freq)
                and isinstance(action_freq[i], list)
                and len(action_freq[i]) == M
            ):
                empirical = self._normalize([float(x) for x in action_freq[i]])
            else:
                empirical = uniform[:]

            long_coop = self._mean_coop(row, coop_vals, window=None)
            recent_coop = self._mean_coop(row, coop_vals, window=min(3, len(row)) if row else None)
            last_action = row[0] if row else None
            switch_rate = self._switch_rate(row)
            persistence = 1.0 - switch_rate
            trend = recent_coop - long_coop

            reciprocity_terms: List[float] = []
            if len(row) >= 2:
                for t in range(len(row) - 1):
                    my_action = row[t]
                    if not (isinstance(my_action, int) and 0 <= my_action < M):
                        continue
                    my_coop = coop_vals[my_action]

                    others_prev: List[float] = []
                    for j in range(N):
                        if j == i:
                            continue
                        other_row = history[j]
                        if len(other_row) > (t + 1):
                            a_prev = other_row[t + 1]
                            if isinstance(a_prev, int) and 0 <= a_prev < M:
                                others_prev.append(coop_vals[a_prev])

                    if others_prev:
                        mean_prev = sum(others_prev) / len(others_prev)
                        reciprocity_terms.append(1.0 - abs(my_coop - mean_prev))

            reciprocity_index = (
                float(sum(reciprocity_terms) / len(reciprocity_terms))
                if reciprocity_terms
                else 0.5
            )

            if not row:
                style = "unknown"
            elif long_coop >= 0.80 and persistence >= 0.70:
                style = "cooperative_sticky"
            elif long_coop <= 0.20 and persistence >= 0.70:
                style = "defective_sticky"
            elif reciprocity_index >= 0.72 and 0.20 < long_coop < 0.80:
                style = "reciprocal"
            elif trend >= 0.12:
                style = "improving"
            elif trend <= -0.12:
                style = "hardening"
            else:
                style = "mixed"

            others_recent: List[float] = []
            for j in range(N):
                if j == i:
                    continue
                other_row = history[j]
                if other_row:
                    a0 = other_row[0]
                    if isinstance(a0, int) and 0 <= a0 < M:
                        others_recent.append(coop_vals[a0])
            ref_recent = float(sum(others_recent) / len(others_recent)) if others_recent else 0.5

            if style == "unknown":
                target_coop = 0.5
            elif style == "cooperative_sticky":
                target_coop = max(0.80, 0.5 * recent_coop + 0.5 * long_coop)
            elif style == "defective_sticky":
                target_coop = min(0.20, 0.5 * recent_coop + 0.5 * long_coop)
            elif style == "reciprocal":
                target_coop = 0.65 * ref_recent + 0.35 * recent_coop
            elif style == "improving":
                target_coop = recent_coop + 0.10
            elif style == "hardening":
                target_coop = recent_coop - 0.10
            else:
                target_coop = 0.50 * recent_coop + 0.50 * long_coop

            target_coop = self._clamp(target_coop)

            sharpness = 9.0 if persistence >= 0.75 else 6.0
            target_dist = self._soft_dist_from_target(target_coop, coop_vals, sharpness=sharpness)

            if isinstance(last_action, int) and 0 <= last_action < M:
                last_onehot = [0.0] * M
                last_onehot[last_action] = 1.0
            else:
                last_onehot = uniform[:]

            if not row:
                pred = target_dist
            elif persistence >= 0.75:
                pred = self._weighted_average([
                    (empirical, 0.30),
                    (target_dist, 0.30),
                    (last_onehot, 0.40),
                ])
            else:
                pred = self._weighted_average([
                    (empirical, 0.55),
                    (target_dist, 0.45),
                ])

            if not pred:
                pred = uniform[:]

            pred = self._normalize(pred)
            top_action = self._argmax(pred)

            opponent_action_probs[str(i)] = pred
            opponent_style_summary[str(i)] = {
                "style": style,
                "recent_coop": round(float(recent_coop), 3),
                "long_coop": round(float(long_coop), 3),
                "switch_rate": round(float(switch_rate), 3),
                "reciprocity_index": round(float(reciprocity_index), 3),
                "predicted_target_coop": round(float(target_coop), 3),
                "predicted_top_action": int(top_action),
                "predicted_top_prob": round(float(pred[top_action]), 3),
            }

        state["opponent_action_probs"] = opponent_action_probs
        state["opponent_style_summary"] = opponent_style_summary
        state["opponent_model_mode"] = "deterministic_style_v1"
        return state