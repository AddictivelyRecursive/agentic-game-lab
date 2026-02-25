"""
ipd.env package

Purpose:
- Implements the game environment (simulator + rules) for an N-player repeated game
  with graded cooperation actions, perception noise, q3-global payoff drift, and global streaks.

This package is intentionally model-agnostic:
- It does NOT depend on any LLM or agent framework.
- Agents interact via a minimal callable interface defined in simulator.py.
"""