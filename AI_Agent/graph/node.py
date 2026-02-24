from abc import ABC, abstractmethod
from typing import Dict, Any


class Node(ABC):
    """
    Abstract base class for all graph nodes.

    Each node represents a single processing step in the AI agent's
    decision pipeline (e.g., feature building, payoff computation,
    opponent modeling, decision making, validation).

    All nodes must implement the `run` method.
    """

    def __init__(self, name: str):
        """
        Initialize a graph node.

        Parameters
        ----------
        name : str
            Unique identifier of the node within the graph.
        """
        self.name = name

    @abstractmethod
    def run(self, state: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the node's logic.

        Parameters
        ----------
        state : Dict[str, Any]
            Mutable state dictionary shared across all nodes.
            This contains game specification, features, intermediate
            computations, and final decisions.

        context : Dict[str, Any]
            Execution context containing shared services such as:
                - LLM client
                - Prompt templates
                - Configuration parameters

        Returns
        -------
        Dict[str, Any]
            Updated state dictionary.
        """
        pass