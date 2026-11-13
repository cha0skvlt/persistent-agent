"""Tests for web intent detection."""

from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

from src.tools.web_intent import (
    extract_http_urls,
    is_web_search_intent,
    is_web_search_intent_with_context,
    prior_human_texts,
    resolve_page_url,
    suggest_web_search_query,
    wants_url_fetch,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    "text",
    [
        "google the copper exchange rate",
        "find on the web the latest news",
        "agent, urgently search the internet for Hitler song lyrics as Kalki",
        "find the song lyrics carnival ended dear sir",
        "google what advaita means",
        "search the web for rust async",
        "google this error message",
        "https://2ch.org/b/ open and quote the first post by Abu",
        "go here https://example.com/page",
        "google the dollar rate",
        "Who are you — and also google what LangGraph is",
        "look it up online",
    ],
)
def test_web_intent_positive(text):
    assert is_web_search_intent(text) is True


@pytest.mark.unit
@pytest.mark.parametrize(
    "text",
    [
        "hello",
        "who is USER",
        "tell me from memory",
    ],
)
def test_web_intent_negative(text):
    assert is_web_search_intent(text) is False


@pytest.mark.unit
def test_resolve_page_url_joins_unrelated_path():
    assert (
        resolve_page_url("/other", ["https://2ch.org/b/"]) == "https://2ch.org/b/other"
    )


@pytest.mark.unit
def test_prior_human_texts_multimodal():
    from src.tools.web_intent import prior_human_texts

    msgs = [HumanMessage(content=[{"text": "https://example.com/a"}])]
    assert "example.com" in prior_human_texts(msgs)[0]


@pytest.mark.unit
def test_fetch_intent_needs_prior_url_in_thread():
    msgs = [
        HumanMessage(content="https://2ch.org/b/"),
        HumanMessage(content="quote the first post"),
    ]
    assert is_web_search_intent_with_context("quote the first post", msgs) is True
    assert (
        resolve_page_url(
            "quote the first post", prior_human_texts(msgs, exclude_last=True)
        )
        == "https://2ch.org/b/"
    )


@pytest.mark.unit
def test_is_web_search_intent_empty():
    assert is_web_search_intent("") is False
    assert is_web_search_intent_with_context("", []) is False


@pytest.mark.unit
def test_wants_url_fetch_detects_inline_url():
    assert wants_url_fetch("see https://example.com/x") is True


@pytest.mark.unit
def test_prior_human_texts_skips_non_human():
    from langchain_core.messages import AIMessage

    from src.tools.web_intent import prior_human_texts

    msgs = [AIMessage(content="bot"), HumanMessage(content="https://x.test/a")]
    assert prior_human_texts(msgs) == ["https://x.test/a"]


@pytest.mark.unit
def test_resolve_page_url_returns_none_without_base():
    assert resolve_page_url("/b", None) is None
    assert resolve_page_url("/b", ["no links"]) is None


@pytest.mark.unit
def test_relative_path_with_prior_url():
    msgs = [
        HumanMessage(content="https://2ch.org/b/ go here and quote Abu's post"),
        HumanMessage(content="/b"),
    ]
    assert is_web_search_intent("/b") is False
    assert is_web_search_intent_with_context("/b", msgs) is True
    assert resolve_page_url("/b", ["https://2ch.org/b/"]) == "https://2ch.org/b"
    assert wants_url_fetch("/b", ["https://2ch.org/b/"]) is True


@pytest.mark.unit
def test_extract_http_urls_strips_trailing_punct():
    urls = extract_http_urls("see https://2ch.org/b/. done")
    assert urls == ["https://2ch.org/b/"]


@pytest.mark.unit
def test_suggest_web_search_query():
    assert "copper" in suggest_web_search_query("google the copper exchange rate")
    assert "Abu" in suggest_web_search_query("google who is Abu")
    q = suggest_web_search_query("agent, google bitard slang and main memes like YOBA")
    assert "bitard" in q.lower() and "slang" in q.lower()
    assert "agent" not in q.lower()
    assert suggest_web_search_query("  ") == ""
    q_url = suggest_web_search_query("https://2ch.org/b/ open and quote the first post")
    assert "2ch.org" not in q_url
    assert "quote" in q_url.lower() or "post" in q_url.lower()
    q_lyrics = suggest_web_search_query(
        "now find the song lyrics carnival ended dear sir dear lady, "
        "also david tibet"
    )
    assert "lyrics" in q_lyrics.lower() or "carnival" in q_lyrics.lower()
    assert "david" in q_lyrics.lower() or "tibet" in q_lyrics.lower()
    q_combo = suggest_web_search_query("Who are you — and also google LangGraph")
    assert "langgraph" in q_combo.lower()


@pytest.mark.unit
def test_trim_search_query_cap(monkeypatch):
    from src.config import settings

    monkeypatch.setattr(settings, "web_search_query_max_chars", 40, raising=False)
    q = suggest_web_search_query("google " + "word " * 20)
    assert len(q) <= 40
    monkeypatch.setattr(settings, "web_search_query_max_chars", "bad", raising=False)
    q2 = suggest_web_search_query("google " + "x" * 30)
    assert len(q2) <= 160

    from src.tools.web_intent import _trim_search_query

    assert _trim_search_query("") == ""
    assert _trim_search_query("   ") == ""


@pytest.mark.unit
def test_search_results_explicit_max_retries(monkeypatch):
    from src.tools import web_core as wc

    monkeypatch.setattr(wc.settings, "web_search_enabled", True, raising=False)
    calls: list[int] = []

    def _search(q, max_results=5):
        calls.append(1)
        return [{"title": "T", "href": "https://x", "body": "b"}]

    monkeypatch.setattr(wc, "_web_text_search", _search)
    wc.search_web_results("google test query here", max_retries=2)
    assert calls


@pytest.mark.unit
def test_search_results_truncates_500(monkeypatch):
    from src.tools import web_core as wc
    from src.tools import web_intent as wi

    monkeypatch.setattr(wc.settings, "web_search_enabled", True, raising=False)
    monkeypatch.setattr(wi, "suggest_web_search_query", lambda _q: "z" * 501)
    seen: list[str] = []

    def _search(q, max_results=5):
        seen.append(q)
        return []

    monkeypatch.setattr(wc, "_web_text_search", _search)
    wc.search_web_results("ignored")
    assert seen and len(seen[0]) == 500
