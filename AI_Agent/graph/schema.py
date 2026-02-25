from typing import Dict, Callable


def build_graph_schema() -> Dict[str, Dict[str, str]]:
    """
    Define directed transitions between nodes.

    Returns
    -------
    Dict[str, Dict[str, str]]
        A mapping of:
            {
                current_node_name: {
                    "next": next_node_name
                }
            }

    Conditional routing (e.g., skip noise if p == 0)
    will be handled inside GraphRunner using state inspection.
    """

    return {
        "N0_ParseTurn": {"next": "N1_BuildFeatures"},
        "N1_BuildFeatures": {"next": "N2_ComputePayoff"},
        "N2_ComputePayoff": {"next": "N3_ComputeNoise"},
        "N3_ComputeNoise": {"next": "N4_OpponentModel"},
        "N4_OpponentModel": {"next": "N5_RankActions"},
        "N5_RankActions": {"next": "N6_DecisionPolicy"},
        "N6_DecisionPolicy": {"next": "N7_ValidateDecision"},
        "N7_ValidateDecision": {"next": "N8_RepairOrFinalize"},
        "N8_RepairOrFinalize": {"next": "N10_ReturnAction"},
        "N9_FallbackDecision": {"next": "N10_ReturnAction"},
        "N10_ReturnAction": {"next": None}
    }