from ..node import Node


class N7_ValidateDecision(Node):
    """
    Deterministic validator.

    Validates only:
    - decision exists
    - key "a" exists
    - "a" is an int
    - 0 <= a < M

    "reason" and "confidence" are OPTIONAL and never cause failure.
    """

    def __init__(self):
        super().__init__("N7_ValidateDecision")

    def run(self, state, context):
        decision = state.get("decision")

        if decision is None:
            state["is_valid"] = False
            state["validation_error"] = "Decision missing"
            return state

        if "a" not in decision:
            state["is_valid"] = False
            state["validation_error"] = "Missing key 'a'"
            return state

        a = decision["a"]

        # Common robustness: allow numeric strings like "3"
        if isinstance(a, str):
            if a.strip().isdigit():
                a = int(a.strip())
                state["decision"]["a"] = a
            else:
                state["is_valid"] = False
                state["validation_error"] = "Action 'a' is not parseable as int"
                return state

        if not isinstance(a, int):
            state["is_valid"] = False
            state["validation_error"] = "Action 'a' is not an int"
            return state

        if not (0 <= a < state["M"]):
            state["is_valid"] = False
            state["validation_error"] = f"Action out of range: {a}"
            return state

        state["is_valid"] = True
        state.pop("validation_error", None)
        return state