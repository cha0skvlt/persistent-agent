"""LangGraph Studio / ``langgraph dev`` entrypoint.

LangGraph API owns checkpoint persistence — do not pass PostgresSaver/MemorySaver here.
"""

from src.graph.workflow import build_app

graph = build_app(for_studio=True)
