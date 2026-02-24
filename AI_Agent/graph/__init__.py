"""
Graph execution engine for the AI Agent.

This module defines the workflow orchestration layer that connects
all decision-making nodes in a directed execution graph.

The graph runner is responsible for:
- Executing nodes in order
- Handling conditional skips (e.g., p == 0)
- Managing retries and fallback logic
- Maintaining execution trace for logging

Nodes themselves contain task-specific logic.
"""

from .node import Node
from .runner import GraphRunner
from .schema import build_graph_schema