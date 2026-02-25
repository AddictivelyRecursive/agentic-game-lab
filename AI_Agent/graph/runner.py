from typing import Dict, Any
from .schema import build_graph_schema


class GraphRunner:
    """
    Orchestrates execution of graph nodes in a controlled manner.

    Responsibilities:
    - Sequential node execution
    - Conditional skipping (e.g., noise when p == 0)
    - Retry logic for invalid LLM outputs
    - Fallback after max retries
    - Execution trace logging
    """

    def __init__(self, nodes: Dict[str, Any], max_retries: int = 3):
        """
        Parameters
        ----------
        nodes : Dict[str, Node]
            Mapping of node_name -> Node instance

        max_retries : int
            Maximum allowed LLM retries before fallback decision.
        """
        self.nodes = nodes
        self.schema = build_graph_schema()
        self.max_retries = max_retries

    def run(self, initial_state: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        state = initial_state
        state["trace"] = []
        state["retries"] = 0

        current_node = "N0_ParseTurn"

        while current_node is not None:
            node = self.nodes[current_node]

            # Conditional Skip: Skip noise node if p == 0
            if current_node == "N3_ComputeNoise":
                if state["game_parameters"]["perception_noise_p"] == 0:
                    state["trace"].append({
                        "node": current_node,
                        "status": "skipped",
                        "reason": "p == 0"
                    })
                    current_node = self.schema[current_node]["next"]
                    continue

            # Guard: never allow ReturnAction with missing decision
            if current_node == "N10_ReturnAction":
                if state.get("decision") is None or "a" not in state["decision"]:
                    raise RuntimeError(
                        "GraphRunner: reached N10_ReturnAction without a valid decision. "
                        "Check validation/repair/fallback routing."
                    )

            # Execute node
            state = node.run(state, context)
            state["trace"].append({"node": current_node, "status": "executed"})

            # Routing logic around validation / repair / fallback
            if current_node == "N7_ValidateDecision":
                if not state.get("is_valid", False):
                    if state["retries"] < self.max_retries:
                        state["retries"] += 1
                        current_node = "N8_RepairOrFinalize"
                    else:
                        current_node = "N9_FallbackDecision"
                    continue
                else:
                    current_node = "N10_ReturnAction"
                    continue

            if current_node == "N8_RepairOrFinalize":
                # After repair attempt, validate again
                current_node = "N7_ValidateDecision"
                continue

            if current_node == "N9_FallbackDecision":
                # Fallback sets a deterministic decision; validate once (optional but clean)
                current_node = "N7_ValidateDecision"
                continue

            # Default: follow schema
            current_node = self.schema[current_node]["next"]

        return state