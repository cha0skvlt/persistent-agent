"""Turn router — retrieval plan per USER message."""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from src.graph.turn_router import TurnPlan, plan_turn


def test_plan_web_only():
    plan = plan_turn("google LangGraph", [])
    assert plan.web_intent
    assert plan.web_search_query
    assert "LangGraph" in (plan.web_search_query or "")


def test_plan_with_thread_context():
    msgs = [
        HumanMessage(content="tell me about LangGraph"),
        HumanMessage(content="yes, google it"),
    ]
    plan = plan_turn("yes, google it", msgs)
    assert plan.web_intent


def test_plan_plain_turn():
    plan = plan_turn("hello", [])
    assert not plan.web_intent
    assert plan.web_search_query is None


def test_web_fast_path_only_when_skip_memory(monkeypatch):
    from src.config import settings

    monkeypatch.setattr(settings, "web_search_prefetch_on_intent", True, raising=False)
    monkeypatch.setattr(settings, "web_skip_memory_on_intent", False, raising=False)
    plan = plan_turn("google LangGraph", [])
    assert plan.web_intent
    assert plan.web_fast_path is False

    monkeypatch.setattr(settings, "web_skip_memory_on_intent", True, raising=False)
    plan_skip = plan_turn("google LangGraph", [])
    assert plan_skip.web_fast_path is True


def test_turn_plan_roundtrip_dict():
    plan = plan_turn("google the dollar rate", [])
    restored = TurnPlan.from_dict(plan.to_dict())
    assert restored is not None
    assert restored.web_intent is plan.web_intent
    assert restored.user_text == plan.user_text
    assert restored.web_search_query == plan.web_search_query
