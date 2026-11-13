"""Unit tests for Mind Hub cross-channel timeline."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.config import settings
from src.mind import hub as mind_mod


@pytest.mark.unit
def test_record_user_in_skips_solitary_threads(monkeypatch):
    calls: list[tuple] = []

    def _record(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(settings, "mind_hub_enabled", True, raising=False)
    monkeypatch.setattr(mind_mod, "_record", _record)
    mind_mod.record_user_in(
        text="hello",
        thread_id="solitary-deadbeef",
        channel="solitary",
    )
    assert calls == []


@pytest.mark.unit
def test_record_agent_out_solitary_kind(monkeypatch):
    captured: list[tuple] = []

    def _record(**kwargs):
        captured.append(kwargs)

    monkeypatch.setattr(settings, "mind_hub_enabled", True, raising=False)
    monkeypatch.setattr(mind_mod, "_record", _record)
    mind_mod.record_agent_out(
        reply="internal thought",
        thread_id="solitary-abc12345",
        channel="repl",
    )
    assert captured[0]["kind"] == "solitary_pass"
    assert captured[0]["channel"] == "solitary"


@pytest.mark.unit
def test_record_agent_out_skips_intent_threads(monkeypatch):
    """Intent visible reply is not USER delivery — do not fake agent_out."""
    captured: list[tuple] = []

    def _record(**kwargs):
        captured.append(kwargs)

    monkeypatch.setattr(settings, "mind_hub_enabled", True, raising=False)
    monkeypatch.setattr(mind_mod, "_record", _record)
    mind_mod.record_agent_out(
        reply="Intent completed on time…",
        thread_id="intent-06c02e48",
        channel="repl",
    )
    assert captured == []


@pytest.mark.unit
def test_fetch_mind_timeline_formats_newest_last(monkeypatch):
    monkeypatch.setattr(settings, "mind_hub_enabled", True, raising=False)
    monkeypatch.setattr(settings, "mind_hub_max_events", 10, raising=False)
    monkeypatch.setattr(settings, "agent_tz", "UTC", raising=False)

    t1 = datetime(2026, 7, 4, 10, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 7, 4, 11, 0, tzinfo=timezone.utc)
    rows = [
        ("user_in", "telegram", "first", t2),
        ("agent_out", "telegram", "second", t1),
    ]

    class _Cur:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, *_a, **_k):
            return self

        def fetchall(self):
            return rows

    conn = MagicMock()
    conn.cursor.return_value = _Cur()
    pool = MagicMock()
    pool.connection.return_value.__enter__ = lambda *_: conn
    pool.connection.return_value.__exit__ = lambda *_: False

    with patch("src.db.pool.get_pool", return_value=pool):
        lines = mind_mod.fetch_mind_timeline(limit=2)

    assert len(lines) == 2
    assert "AGENT OUT" in lines[0]
    assert "USER IN" in lines[1]


@pytest.mark.unit
def test_build_mind_hub_sys_block_empty():
    assert mind_mod.build_mind_hub_sys_block([]) is None


@pytest.mark.unit
def test_build_mind_hub_sys_block_non_empty():
    block = mind_mod.build_mind_hub_sys_block(
        ["[2026-07-04 10:00] USER IN (telegram): hi"]
    )
    assert block is not None
    assert "MIND HUB" in block
    assert "USER IN" in block


@pytest.mark.unit
def test_record_outreach_queued(monkeypatch):
    captured: list[tuple] = []

    def _record(**kwargs):
        captured.append(kwargs)

    monkeypatch.setattr(settings, "mind_hub_enabled", True, raising=False)
    monkeypatch.setattr(mind_mod, "_record", _record)
    mind_mod.record_outreach_queued(draft="hello there")
    assert captured[0]["kind"] == "outreach_queued"
    assert "hello there" in captured[0]["content"]


@pytest.mark.unit
def test_primary_live_thread_id_from_owner_chat(monkeypatch):
    monkeypatch.setattr(settings, "telegram_owner_chat_id", "12345", raising=False)
    assert mind_mod.primary_live_thread_id() == "telegram_12345"
