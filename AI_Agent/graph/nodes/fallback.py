from ..node import Node


class N9_FallbackDecision(Node):
    """
    Deterministic fallback if LLM fails after max retries (strict research mode).

    Assumptions:
    - N5_RankActions MUST have produced a non-empty `state["candidates"]` list
      with length M (one entry per action).
    - If candidates are missing/empty, this indicates a pipeline bug and we
      fail fast to surface the issue during experiments.

    Behavior:
    - Choose the highest expected-utility action from `state["candidates"]`.
    - Attach a short reason + confidence for trace/paper analysis.
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

        best = candidates[0]
        if "action" not in best:
            raise RuntimeError(
                "FallbackDecision: candidate missing key 'action'. "
                "Candidates must contain {'action': int, 'EU': float, ...}."
            )

        state["decision"] = {
            "a": best["action"],
            "reason": "Fallback: LLM output invalid after retries; choosing highest expected-utility action.",
            "confidence": 0.90
        }
        state["is_valid"] = True
        return state