# POST-HOC / EXPLORATORY: exact saved v2 parser and validator.
# Copied from src/AI_Agent/agent/pipeline.py, with no inference code.
# Function source SHA256: 04769c445f4f13528cc0789483d6fb13202e257c7f2538067b727761a348e322
import json

def extract_decision_json(response):
    """Read a unique decision object after reasoning; never repair truncated JSON."""
    text = response
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    elif "<think>" in text:
        raise ValueError("Unterminated reasoning block")
    decoder = json.JSONDecoder()
    candidates = []
    offset = 0
    while offset < len(text):
        start = text.find("{", offset)
        if start < 0:
            break
        try:
            value, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            offset = start + 1
            continue
        if isinstance(value, dict) and "a" in value:
            candidates.append(value)
        offset = end
    if len(candidates) != 1:
        raise ValueError("Expected exactly one complete decision JSON object")
    return candidates[0]


def validate_decision(state):
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

    if isinstance(a, bool) or not isinstance(a, int):
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
