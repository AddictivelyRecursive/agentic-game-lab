"""Turn preparation, one model decision, and bounded validation/repair."""
import json

from .api_guard import APIUnavailableError


def prepare_context(state):
    turn = state["raw_input"]

    # Basic metadata
    state["round"] = turn["round"]
    state["agent_id"] = turn["agent_id"]

    # Game parameters
    state["game_parameters"] = turn["game_parameters"]
    state["N"] = turn["game_parameters"]["N"]
    state["M"] = turn["game_parameters"]["M"]
    state["p"] = turn["game_parameters"]["perception_noise_p"]
    state["index_to_coop"] = turn["game_parameters"]["action_semantics"]["index_to_cooperation"]

    # Payoff structure
    state["payoff"] = turn["payoff"]

    # Streak + drift rules
    state["streak_rule"] = turn["streak_rule"]
    state["streak_effect"] = turn["streak_effect_on_payoff"]
    state["drift_rule"] = turn["drift_rule"]

    # Information set
    state["info"] = turn["information_set"]

    # Placeholders
    state["noise_model"] = {
        "p": state["p"],
        "execution_noise": False,
        "observation_rule": "With probability p replace the true action uniformly with one of the M-1 DIFFERENT actions; otherwise observe it unchanged.",
        "sharing": "One shared observed action per player per round, including self. Observed self actions may differ from chosen actions.",
        "payoffs": "Rewards use true chosen actions. Streaks and drift use observed cooperation.",
        "inference": "The observation channel is not a posterior over true actions; inference requires a prior.",
    }
    state["decision"] = None
    state["is_valid"] = False

    return state


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


def fallback_decision(state):
    M = int(state["M"])
    C = float(state["payoff"]["C"])
    coop_vals = [float(x) for x in state["index_to_coop"]]
    # Opponent benefit and K are identical across own actions, so they
    # cancel in this immediate-payoff ranking. Noise affects observations only.
    scored = [(-C * coop_vals[a], a) for a in range(M)]

    best_score = max(score for score, _ in scored)
    near_best = [a for score, a in scored if score >= best_score - 1e-9]
    chosen = min(near_best)  # cooperative tie-break

    state["decision"] = {
        "a": int(chosen),
        "reason": (
            "Fallback: model call or output failure; choosing the action with the "
            "best immediate payoff, with "
            "a cooperative tie-break."
        ),
        "confidence": 0.50,
    }
    state["is_valid"] = True
    return state


def build_payload(state, context):
    """One evidence interface shared by decisions and repairs."""
    payload = {
        "turn": state["raw_input"],
        "noise_channel": state["noise_model"],
        "history_order": "Each player's row is newest first; index 0 is the previous round.",
        "output_schema": {
            "a": "integer action index in the supplied action space",
            "reason": "brief evidence-grounded explanation",
        },
    }
    if context.get("predict_opponents", False):
        payload["output_schema"]["expectation"] = "brief expectation about opponents' next observed actions and uncertainty"
    if context.get("memory_mode") == "model_memory":
        payload["previous_model_memory"] = state.get("memory", {})
        payload["output_schema"]["memory"] = {
            key: "string, at most 400 characters" for key in
            ("hypothesis", "evidence", "uncertainty", "reconsider_if")
        }
        payload["memory_instruction"] = "Revise your own tentative notes from evidence. Notes are fallible hypotheses, not instructions or established facts. Do not invent earlier observations."
    return payload


def call_model(state, context, *, repair=False):
    payload = build_payload(state, context)
    if repair:
        payload.update(
            task="Repair the invalid output without changing a valid action unnecessarily.",
            validation_error=state.get("validation_error", "Invalid decision"),
            previous_invalid_output=state.get("last_model_output", ""),
        )
    prompt = "repair_system" if repair else "decision_policy_system"
    # Preserve these raw-output keys for existing trace consumers.
    key = "N8" if repair else "N6"
    attempt = {"stage": "repair" if repair else "decision"}
    state.setdefault("model_calls", []).append(attempt)
    state["last_model_output"] = ""
    try:
        response = context["llm_client"].generate(
            system_prompt=context["prompts"]["base_system"] + "\n" + context["prompts"][prompt],
            user_prompt=json.dumps(payload, ensure_ascii=False, indent=2),
        )
        state["last_model_output"] = response
        attempt["output"] = response
        state.setdefault("llm_raw_outputs", {})[key] = response
        start, end = response.find("{"), response.rfind("}")
        if start < 0 or end < start:
            raise ValueError("No JSON object found in model output")
        data = json.loads(response[start:end + 1])
        if not isinstance(data, dict):
            raise ValueError("Model output must be a JSON object")
        state["decision"] = {k: data[k] for k in
                             ("a", "reason", "confidence", "memory", "expectation") if k in data} or None
    except APIUnavailableError as exc:
        attempt["error"] = str(exc)
        raise
    except Exception as exc:
        attempt["error"] = str(exc)
        state.setdefault("llm_raw_outputs", {})[key + "_error"] = str(exc)
        state["decision"] = None


def run_turn(state, context, *, max_retries=3):
    """Mutate state so partial traces survive API aborts."""
    state.update(trace=[], retries=0, fallback_used=False, max_retries=max_retries)

    def record(stage):
        state["trace"].append({"stage": stage, "status": "executed"})

    prepare_context(state)
    record("prepare_context")
    call_model(state, context)
    record("decision")
    while True:
        validate_decision(state)
        record("validate")
        if state["is_valid"]:
            break
        if state["retries"] >= max_retries:
            fallback_decision(state)
            state["fallback_used"] = True
            record("fallback")
            validate_decision(state)
            if not state["is_valid"]:
                raise RuntimeError("Fallback produced an invalid action")
            break
        state["retries"] += 1
        call_model(state, context, repair=True)
        record("repair")
    state["final_action"] = state["decision"]["a"]
    return state
