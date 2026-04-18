from ..node import Node


class N9_FallbackDecision(Node):
    """
    Deterministic fallback if LLM fails after max retries.

    Important change:
    - We no longer describe fallback as "highest expected-utility action".
    - We use the strategic ranking produced by N5.
    - If several actions are nearly tied, prefer the more cooperative one.
    """

    def __init__(self):
        super().__init__("N9_FallbackDecision")

    def run(self, state, context):
        candidates = state.get("candidates", None)
        if candidates is None:
            raise RuntimeError(
                "FallbackDecision: `state['candidates']` missing. "
                "This indicates RankActions did not run or did not set candidates."
            )
        if not isinstance(candidates, list):
            raise RuntimeError(
                f"FallbackDecision: `state['candidates']` must be a list, got {type(candidates)}."
            )
        if len(candidates) == 0:
            raise RuntimeError(
                "FallbackDecision: `state['candidates']` is empty. "
                "This should never happen if RankActions is correct."
            )

        score_key = "strategic_score" if "strategic_score" in candidates[0] else "EU"
        top_score = float(candidates[0][score_key])

        # Near-tie band: prefer more cooperative action if strategic quality is basically the same.
        tolerance = 0.05 if score_key == "strategic_score" else 1e-9
        near_top = [
            c for c in candidates
            if float(c.get(score_key, top_score)) >= (top_score - tolerance)
        ]

        chosen = min(near_top, key=lambda c: c["action"])  # lower index = more cooperative
        fallback_action = state.get("fallback_action", chosen["action"])

        state["decision"] = {
            "a": int(fallback_action),
            "reason": (
                "Fallback: LLM output invalid after retries; "
                "choosing top strategic action, with a cooperative tie-break among near-equals."
            ),
            "confidence": 0.90,
        }
        state["is_valid"] = True
        return state