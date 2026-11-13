"""persistent-agent python package root."""

# LangGraph checkpoint serde must patch Reviver before any langgraph import
# (e.g. ``langgraph.errors`` in main.py pulls encrypted → jsonplus).
import src.graph.checkpoint_serde  # noqa: F401
