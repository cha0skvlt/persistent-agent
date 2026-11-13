"""Retrieval web-intent branch coverage."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import HumanMessage

from src.config import settings
from src.graph import retrieval_node as rn
from src.tools.web_prefetch import WEB_FETCH_MARKER


@pytest.mark.unit
def test_retrieve_skips_memory_on_web_intent(monkeypatch):
    embed = MagicMock()
    monkeypatch.setattr(rn, "_embed_query", embed, raising=True)
    monkeypatch.setattr(settings, "web_skip_memory_on_intent", True, raising=False)
    monkeypatch.setattr(settings, "web_search_prefetch_on_intent", True, raising=False)
    monkeypatch.setattr(
        rn,
        "prefetch_web_for_message",
        lambda msg, prior_human=None: ["[WEB] search_web: query='q'\nresults"],
    )

    out = rn.retrieve_memory(
        {
            "messages": [HumanMessage(content="search the web for latest ollama news")],
        }
    )
    embed.assert_not_called()
    mems = out.get("retrieved_memories") or []
    assert any(str(m).startswith("[WEB] search_web:") for m in mems)
    assert out.get("retrieved_self_reflections") == []


@pytest.mark.unit
def test_retrieve_web_hint_when_prefetch_disabled(monkeypatch):
    monkeypatch.setattr(settings, "web_skip_memory_on_intent", True, raising=False)
    monkeypatch.setattr(settings, "web_search_prefetch_on_intent", False, raising=False)
    out = rn.retrieve_memory(
        {
            "messages": [HumanMessage(content="google ollama")],
        }
    )
    mems = out.get("retrieved_memories") or []
    assert any("Live web request" in str(m) for m in mems)


@pytest.mark.unit
def test_retrieve_prefetches_when_skip_disabled(monkeypatch):
    monkeypatch.setattr(settings, "web_skip_memory_on_intent", False, raising=False)
    monkeypatch.setattr(settings, "web_search_prefetch_on_intent", True, raising=False)
    embed_calls: list[str] = []

    def _embed(mem, q, **k):
        embed_calls.append(q)
        return [0.1, 0.2]

    monkeypatch.setattr(rn, "_embed_query", _embed)
    monkeypatch.setattr(
        rn,
        "_memory_rows_for_rerank",
        lambda *a, **k: [
            ("mem-web", "USER drinks dark roast every morning", 0.91, 0.5),
        ],
    )
    monkeypatch.setattr(
        rn,
        "_fetch_rerank_meta",
        lambda *a, **k: {"mem-web": ("semantic", None)},
    )
    monkeypatch.setattr(rn, "_fetch_self_reflections", lambda *a, **k: ([], []))
    monkeypatch.setattr(
        rn,
        "prefetch_web_for_message",
        lambda msg, prior_human=None: ["[WEB] search_web: query='q'\nresults"],
    )

    out = rn.retrieve_memory(
        {
            "messages": [HumanMessage(content="google ollama")],
        }
    )
    mems = out.get("retrieved_memories") or []
    assert embed_calls
    assert any(str(m).startswith("[WEB] search_web:") for m in mems)
    assert any("[MEM]" in str(m) and "dark roast" in str(m) for m in mems)
    assert not any("Call search_web" in str(m) for m in mems)


@pytest.mark.unit
def test_retrieve_fetch_hint_when_prefetch_disabled(monkeypatch):
    monkeypatch.setattr(settings, "web_skip_memory_on_intent", True, raising=False)
    monkeypatch.setattr(settings, "web_search_prefetch_on_intent", False, raising=False)
    out = rn.retrieve_memory(
        {
            "messages": [
                HumanMessage(
                    content="https://2ch.org/b/ open and quote the first post by Abu"
                )
            ],
        }
    )
    mems = out.get("retrieved_memories") or []
    assert any("Live page fetch" in str(m) and "fetch_url" in str(m) for m in mems)


@pytest.mark.unit
def test_retrieve_injects_web_hint_when_skip_disabled(monkeypatch):
    monkeypatch.setattr(settings, "web_skip_memory_on_intent", False, raising=False)
    monkeypatch.setattr(settings, "web_search_prefetch_on_intent", False, raising=False)
    monkeypatch.setattr(
        rn,
        "_embed_query",
        lambda mem, q, **k: [0.1, 0.2],
    )
    monkeypatch.setattr(
        rn,
        "_memory_rows_for_rerank",
        lambda *a, **k: [],
    )
    monkeypatch.setattr(rn, "_fetch_self_reflections", lambda *a, **k: ([], []))

    out = rn.retrieve_memory(
        {
            "messages": [HumanMessage(content="search the web for latest ollama news")],
        }
    )
    mems = out.get("retrieved_memories") or []
    assert any(str(m).startswith("[WEB]") for m in mems)


@pytest.mark.unit
def test_retrieve_fetches_url_on_visit(monkeypatch):
    monkeypatch.setattr(settings, "web_skip_memory_on_intent", True, raising=False)
    monkeypatch.setattr(settings, "web_search_prefetch_on_intent", True, raising=False)
    fetched: list[str] = []

    def _prefetch(msg, prior_human=None):
        fetched.append(msg)
        return [f"{WEB_FETCH_MARKER} url='https://2ch.org/b'\nthread"]

    monkeypatch.setattr(rn, "prefetch_web_for_message", _prefetch)
    out = rn.retrieve_memory(
        {
            "messages": [
                HumanMessage(
                    content="https://2ch.org/b/ open and quote the first post by Abu"
                )
            ],
        }
    )
    assert fetched
    mems = out.get("retrieved_memories") or []
    assert "fetch_url" in str(mems[0])
