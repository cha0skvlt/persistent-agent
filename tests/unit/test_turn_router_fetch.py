"""Fetch-intent routing: resolve prior URL or skip mandatory search."""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from src.graph.turn_router import plan_turn


def test_fetch_intent_resolves_prior_url():
    msgs = [
        HumanMessage(content="https://example.com/thread"),
        HumanMessage(content="quote the first post on this page"),
    ]
    plan = plan_turn("quote the first post on this page", msgs)
    assert plan.web_intent
    assert plan.web_search_query is None
    assert plan.web_fetch_url == "https://example.com/thread"
    assert plan.web_fetch_wanted


def test_fetch_intent_without_url_in_thread_not_web_turn():
    msgs = [HumanMessage(content="some context without a link")]
    plan = plan_turn("quote the first post on this page", msgs)
    assert not plan.web_intent
    assert plan.web_fetch_url is None
    assert not plan.web_fetch_wanted
    assert plan.web_search_query is None
