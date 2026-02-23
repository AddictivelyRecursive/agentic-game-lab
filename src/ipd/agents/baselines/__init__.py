"""
agents/baselines package

Purpose:
- Baseline agents for end-to-end testing of env without LLM calls.
"""

from .always import AlwaysCooperate, AlwaysDefect
from .random_agent import RandomAgent
from .graded_tft import GradedTFT