from ..node import Node


class N10_ReturnAction(Node):
    """
    Final node.

    Extracts validated decision and stores it as output.
    """

    def __init__(self):
        super().__init__("N10_ReturnAction")

    def run(self, state, context):

        state["final_action"] = state["decision"]["a"]
        return state