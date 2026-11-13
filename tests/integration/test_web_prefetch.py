"""Tests for web search prefetch on user intent."""

from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

from src.config import settings
from src.graph import retrieval_node as rn
from src.tools.web_prefetch import (
    WEB_FETCH_MARKER,
    WEB_PREFETCH_MARKER,
    auto_tools_from_state,
    prefetch_web_fetch,
    prefetch_web_for_message,
    prefetch_web_search,
    prefetched_tool_name,
    web_prefetch_succeeded,
)


@pytest.mark.unit
def test_prefetch_web_search(monkeypatch):
    monkeypatch.setattr(
        "src.tools.web_core.search_web_results",
        lambda q, max_retries=1: f"Web search: {q}\n1. Hit",
    )
    out = prefetch_web_search("google ollama news")
    assert out[0].startswith(WEB_PREFETCH_MARKER)
    assert "ollama" in out[0].lower()
    assert "1. Hit" in out[0]


@pytest.mark.unit
def test_auto_tools_from_state():
    assert auto_tools_from_state({"retrieved_memories": []}) == []
    good = (
        f"{WEB_PREFETCH_MARKER} query='x'\nWeb search: usd thb\n\n1. Rate\n   32.5 THB"
    )
    assert auto_tools_from_state({"retrieved_memories": [good]}) == ["search_web"]
    assert auto_tools_from_state(
        {"retrieved_memories": [f"{WEB_PREFETCH_MARKER} query='x'\nbody"]}
    ) == ["search_web"]
    assert auto_tools_from_state(
        {"retrieved_memories": [f"{WEB_PREFETCH_MARKER} Web search failed"]}
    ) == ["search_web"]
    fetch_body = f"{WEB_FETCH_MARKER} url='https://x'\n" + ("page text " * 12)
    assert auto_tools_from_state({"retrieved_memories": [fetch_body]}) == ["fetch_url"]


@pytest.mark.unit
def test_prefetch_web_fetch(monkeypatch):
    monkeypatch.setattr(
        "src.tools.web_core.fetch_url_text",
        lambda u: f"Page for {u}",
    )
    out = prefetch_web_fetch("https://2ch.org/b/")
    assert out[0].startswith(WEB_FETCH_MARKER)
    assert "2ch.org" in out[0]


@pytest.mark.unit
@pytest.mark.parametrize(
    "body",
    [
        "empty query",
        "disabled in config",
        "Fetch failed: timeout",
        "URL rejected: private network",
        "HTTP 503 service unavailable",
        "web search failed: upstream",
    ],
)
def test_prefetch_web_fetch_logs_failed_body(monkeypatch, body):
    monkeypatch.setattr("src.tools.web_core.fetch_url_text", lambda u: body)
    out = prefetch_web_fetch("https://example.com/")
    assert body in out[0]


@pytest.mark.unit
def test_prefetch_web_for_message_uses_fetch(monkeypatch):
    monkeypatch.setattr(
        "src.tools.web_prefetch.prefetch_web_fetch",
        lambda url: [f"{WEB_FETCH_MARKER} url={url!r}\n" + ("page " * 20)],
    )
    monkeypatch.setattr(
        "src.tools.web_prefetch.prefetch_web_search",
        lambda msg: [f"{WEB_PREFETCH_MARKER} should not run"],
    )
    out = prefetch_web_for_message(
        "https://2ch.org/b/ open and quote the post by Abu",
    )
    assert out[0].startswith(WEB_FETCH_MARKER)
    assert prefetched_tool_name(out) == "fetch_url"


@pytest.mark.unit
def test_web_prefetch_succeeded_ignores_non_web_memories():
    good = f"{WEB_PREFETCH_MARKER} query='q'\nWeb search: q\n\n1. Hit\n   snippet"
    assert web_prefetch_succeeded(["[MEM] x", good]) is True
    assert (
        web_prefetch_succeeded([f"{WEB_PREFETCH_MARKER} query='q'\nno hits"]) is False
    )


@pytest.mark.unit
def test_auto_tools_reports_prefetch_attempt_on_failed_fetch():
    assert auto_tools_from_state(
        {"retrieved_memories": [f"{WEB_FETCH_MARKER} HTTP 404 for u"]}
    ) == ["fetch_url"]


@pytest.mark.unit
def test_prefetch_web_for_message_falls_back_to_search(monkeypatch):
    monkeypatch.setattr(
        "src.tools.web_prefetch.wants_url_fetch",
        lambda msg, prior_human=None: True,
    )
    monkeypatch.setattr(
        "src.tools.web_prefetch.resolve_page_url",
        lambda msg, prior_human=None: None,
    )
    monkeypatch.setattr(
        "src.tools.web_prefetch.prefetch_web_search",
        lambda msg: [
            f"{WEB_PREFETCH_MARKER} query='x'\nWeb search: x\n\n1. hit\n   ok"
        ],
    )
    out = prefetch_web_for_message("google test")
    assert out[0].startswith(WEB_PREFETCH_MARKER)


@pytest.mark.unit
def test_prefetched_tool_name_search_only():
    assert (
        prefetched_tool_name(
            [f"{WEB_PREFETCH_MARKER} query='q'\nWeb search: q\n\n1. hit\n   ok"]
        )
        == "search_web"
    )


@pytest.mark.unit
def test_prefetch_memory_url_rejected():
    assert (
        web_prefetch_succeeded(
            [f"{WEB_FETCH_MARKER} url='x'\nURL rejected: private network"]
        )
        is False
    )


@pytest.mark.unit
def test_web_prefetch_succeeded_fetch():
    fetch_ok = f"{WEB_FETCH_MARKER} url='x'\n" + ("title ok " * 15)
    assert web_prefetch_succeeded([fetch_ok]) is True
    assert web_prefetch_succeeded([f"{WEB_FETCH_MARKER} Fetch failed: x"]) is False
    assert web_prefetch_succeeded([f"{WEB_FETCH_MARKER} HTTP 404 for u"]) is False


@pytest.mark.unit
def test_prefetch_fetch_empty_and_disabled(monkeypatch):
    assert prefetch_web_fetch("   ")[0].endswith("empty url")
    monkeypatch.setattr(settings, "web_fetch_enabled", False, raising=False)
    out = prefetch_web_fetch("https://2ch.org/b/")
    assert "disabled" in out[0]


@pytest.mark.unit
def test_prefetch_empty_and_disabled(monkeypatch):
    assert prefetch_web_search("   ")[0].endswith("empty query")
    monkeypatch.setattr(settings, "web_search_enabled", False, raising=False)
    out = prefetch_web_search("google x")
    assert "disabled" in out[0]


@pytest.mark.unit
def test_web_prefetch_succeeded():
    assert web_prefetch_succeeded([]) is False
    assert web_prefetch_succeeded([f"{WEB_PREFETCH_MARKER} empty query"]) is False
    assert web_prefetch_succeeded([f"{WEB_PREFETCH_MARKER} disabled"]) is False
    assert web_prefetch_succeeded([f"{WEB_PREFETCH_MARKER} Web search failed"]) is False
    assert (
        web_prefetch_succeeded(
            [f"{WEB_PREFETCH_MARKER} query='x'\nWeb search: x\n\n1. Hit\n   ok"]
        )
        is True
    )


@pytest.mark.unit
def test_retrieve_prefetches_on_web_intent(monkeypatch):
    monkeypatch.setattr(settings, "web_skip_memory_on_intent", True, raising=False)
    monkeypatch.setattr(settings, "web_search_prefetch_on_intent", True, raising=False)
    monkeypatch.setattr(
        rn,
        "prefetch_web_for_message",
        lambda msg, prior_human=None: [f"{WEB_PREFETCH_MARKER} query='q'\nresults"],
    )
    out = rn.retrieve_memory(
        {
            "messages": [HumanMessage(content="google who is Abu")],
        }
    )
    mems = out.get("retrieved_memories") or []
    assert mems[0].startswith(WEB_PREFETCH_MARKER)
