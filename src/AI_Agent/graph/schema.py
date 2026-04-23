from typing import Dict


def build_graph_schema() -> Dict[str, Dict[str, str]]:
    """
    Graph with no hand-engineered opponent model and no pre-ranking stage.
    """
    return {
        "N0_ParseTurn": {"next": "N1_BuildFeatures"},
        "N1_BuildFeatures": {"next": "N2_ComputePayoff"},
        "N2_ComputePayoff": {"next": "N3_ComputeNoise"},
        "N3_ComputeNoise": {"next": "N6_DecisionPolicy"},
        "N6_DecisionPolicy": {"next": "N7_ValidateDecision"},
        "N7_ValidateDecision": {"next": "N8_RepairOrFinalize"},
        "N8_RepairOrFinalize": {"next": "N10_ReturnAction"},
        "N9_FallbackDecision": {"next": "N10_ReturnAction"},
        "N10_ReturnAction": {"next": None},
    }