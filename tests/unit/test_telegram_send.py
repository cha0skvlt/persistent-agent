"""Unit tests for shared Telegram Delivery layer."""

from __future__ import annotations

import pytest

from src.api import telegram_send as ts


class _Resp:
    def __init__(self, ok: bool, status: int = 200, description: str = ""):
        self.status_code = status
        self.text = description
        self._ok = ok
        self._description = description

    def json(self):
        data: dict = {"ok": self._ok}
        if self._description:
            data["description"] = self._description
        return data


@pytest.mark.unit
def test_send_body_sync_html_ok(monkeypatch):
    calls: list[dict] = []

    def _post(url, json=None, timeout=10.0):
        calls.append(json or {})
        return _Resp(True)

    monkeypatch.setattr(ts.httpx, "post", _post)
    assert ts.send_body_sync("tok", "1", "Hello **USER**") is True
    assert calls and calls[0].get("parse_mode") == "HTML"
    assert "Hello" in calls[0]["text"]


@pytest.mark.unit
def test_send_body_sync_empty_after_outreach_sanitize(monkeypatch):
    """Think-only body → no chunks → False (not silent success)."""
    calls: list = []
    monkeypatch.setattr(
        ts.httpx, "post", lambda *a, **k: calls.append(1) or _Resp(True)
    )
    assert (
        ts.send_body_sync("tok", "1", "<think>only thinking</think>", outreach=True)
        is False
    )
    assert calls == []


@pytest.mark.unit
def test_send_body_sync_partial_success_not_total_failure(monkeypatch):
    """After a chunk reaches Telegram, later fail must not look like total miss."""
    calls = {"n": 0}

    def _chunk(*_a, **_k):
        calls["n"] += 1
        return calls["n"] == 1

    monkeypatch.setattr(
        ts,
        "telegram_message_chunks",
        lambda *_a, **_k: [("html1", "p1"), ("html2", "p2")],
    )
    monkeypatch.setattr(ts, "send_chunk_sync", _chunk)
    assert ts.send_body_sync("tok", "1", "long body") is True
    assert calls["n"] == 2


@pytest.mark.unit
def test_send_body_sync_plain_fallback(monkeypatch):
    calls: list[dict] = []

    def _post(url, json=None, timeout=10.0):
        calls.append(json or {})
        if len(calls) == 1:
            return _Resp(False, status=400, description="can't parse entities")
        return _Resp(True)

    monkeypatch.setattr(ts.httpx, "post", _post)
    assert ts.send_body_sync("tok", "1", "**bold** text") is True
    assert calls[0].get("parse_mode") == "HTML"
    assert "parse_mode" not in calls[1]


@pytest.mark.unit
def test_send_chunk_retries_simplified_thinking_html(monkeypatch):
    """Nested <i> in expandable blockquote → retry without italic before plain."""
    calls: list[dict] = []

    def _post(url, json=None, timeout=10.0):
        calls.append(json or {})
        text = (json or {}).get("text", "")
        if "<i>" in text:
            return _Resp(False, status=400, description="can't parse entities")
        return _Resp(True)

    monkeypatch.setattr(ts.httpx, "post", _post)
    html = "<blockquote expandable><i>planning</i></blockquote>\n\n" "visible reply"
    assert ts.send_chunk_sync("tok", "1", html, "planning\n\nvisible reply") is True
    assert len(calls) == 2
    assert "<i>" in calls[0]["text"]
    assert "<blockquote expandable>" in calls[1]["text"]
    assert "<i>" not in calls[1]["text"]
    assert calls[1].get("parse_mode") == "HTML"


@pytest.mark.unit
def test_send_chunk_sync_empty_both():
    assert ts.send_chunk_sync("tok", "1", "", "") is True


@pytest.mark.unit
def test_send_chunk_async_html_ok():
    import asyncio

    posts: list[tuple] = []

    async def _post(method, payload):
        posts.append((method, payload))
        return {"ok": True}

    asyncio.run(ts.send_chunk_async(_post, "1", "<b>hi</b>", "hi"))
    assert posts and posts[0][0] == "sendMessage"
    assert posts[0][1]["parse_mode"] == "HTML"
