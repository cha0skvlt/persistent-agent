"""route_turn node sets turn_plan before retrieval."""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from src.graph.route_turn_node import route_turn_node


def test_route_turn_web():
    out = route_turn_node(
        {
            "messages": [HumanMessage(content="google LangGraph")],
            "current_time": "2026-06-30",
        }
    )
    plan = out["turn_plan"]
    assert plan["web_intent"]
    assert plan["web_search_query"]
    assert "ritual_selftest" not in plan


def test_route_turn_identity_web():
    out = route_turn_node(
        {
            "messages": [HumanMessage(content="Who are you? Google LangGraph")],
            "current_time": "2026-06-30",
        }
    )
    plan = out["turn_plan"]
    assert plan["web_intent"]
    assert "self_query" not in plan
    assert plan["web_search_query"]
