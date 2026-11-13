"""Unit tests for Pulse time-sense SYS block."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.config import settings
from src.graph.context_budget import TRUNCATION_MARKER
from src.pulse import time_sense as ts


@pytest.fixture(autouse=True)
def _time_sense_settings(monkeypatch):
    monkeypatch.setattr(settings, "agent_tz", "UTC", raising=False)
    monkeypatch.setattr(settings, "default_agent_id", "agent", raising=False)
    monkeypatch.setattr(settings, "pulse_enabled", True, raising=False)
    monkeypatch.setattr(settings, "pulse_sys_max_chars", 600, raising=False)


def test_format_human_delta_hours_minutes():
    assert ts.format_human_delta(3 * 3600 + 12 * 60) == "3h 12m"


def test_format_human_delta_days():
    assert ts.format_human_delta(2 * 86400) == "2d"


def test_format_human_delta_hours_only():
    assert ts.format_human_delta(5 * 3600) == "5h"


def test_format_human_delta_minutes_only():
    assert ts.format_human_delta(45 * 60) == "45m"


def test_time_sense_block_empty_returns_none(monkeypatch):
    monkeypatch.setattr(ts, "_delta_line", lambda: None)
    monkeypatch.setattr(ts, "_upcoming_intents_lines", lambda: [])
    assert ts.time_sense_block() is None


def test_time_sense_block_delta_only(monkeypatch):
    monkeypatch.setattr(
        ts, "_delta_line", lambda: "3h 12m since the last USER message."
    )
    monkeypatch.setattr(ts, "_upcoming_intents_lines", lambda: [])
    block = ts.time_sense_block()
    assert block == "3h 12m since the last USER message."


def test_time_sense_block_intents_only(monkeypatch):
    monkeypatch.setattr(ts, "_delta_line", lambda: None)
    monkeypatch.setattr(
        ts,
        "_upcoming_intents_lines",
        lambda: [
            "Upcoming intents:",
            "— 2026-07-09 09:00 ask about X",
        ],
    )
    block = ts.time_sense_block()
    assert block is not None
    assert "Upcoming intents" in block
    assert "ask about X" in block


def test_time_sense_block_truncates_tail(monkeypatch):
    monkeypatch.setattr(ts, "_delta_line", lambda: "1h since the last USER message.")
    long_tail = "— 2026-07-09 09:00 " + ("very long text " * 80)
    monkeypatch.setattr(
        ts,
        "_upcoming_intents_lines",
        lambda: ["Upcoming intents:", long_tail],
    )
    monkeypatch.setattr(settings, "pulse_sys_max_chars", 120, raising=False)
    block = ts.time_sense_block()
    assert block is not None
    assert len(block) <= 120
    assert TRUNCATION_MARKER in block


def test_delta_line_from_idle(monkeypatch):
    monkeypatch.setattr(
        "src.reverie.idle.minutes_since_last_user_turn",
        lambda: 192.0,
    )
    assert ts._delta_line() == "3h 12m since the last USER message."


def test_upcoming_intents_skipped_when_pulse_disabled(monkeypatch):
    monkeypatch.setattr(settings, "pulse_enabled", False, raising=False)
    assert ts._upcoming_intents_lines() == []


def test_upcoming_intents_from_db(monkeypatch):
    run_at = datetime(2026, 7, 9, 9, 0, tzinfo=timezone.utc)

    def _fake_pool():
        conn = MagicMock()
        cur = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cur)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        cur.fetchall.return_value = [(run_at, "ask about X")]
        pool = MagicMock()
        pool.connection.return_value.__enter__ = MagicMock(return_value=conn)
        pool.connection.return_value.__exit__ = MagicMock(return_value=False)
        return pool

    with patch("src.db.pool.get_pool", _fake_pool):
        lines = ts._upcoming_intents_lines()
    assert lines[0] == "Upcoming intents:"
    assert lines[1] == "— 2026-07-09 09:00 ask about X"


def test_upcoming_intents_db_error_returns_empty(monkeypatch):
    def _boom():
        raise RuntimeError("db down")

    with patch("src.db.pool.get_pool", _boom):
        assert ts._upcoming_intents_lines() == []


def test_upcoming_intents_null_run_at(monkeypatch):
    def _fake_pool():
        conn = MagicMock()
        cur = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cur)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        cur.fetchall.return_value = [(None, "no time")]
        pool = MagicMock()
        pool.connection.return_value.__enter__ = MagicMock(return_value=conn)
        pool.connection.return_value.__exit__ = MagicMock(return_value=False)
        return pool

    with patch("src.db.pool.get_pool", _fake_pool):
        lines = ts._upcoming_intents_lines()
    assert lines[-1] == "— ? no time"


def test_pulse_sacrifice_rank_between_web_and_mem():
    from src.graph import context_budget as cb

    # diary/mind first; web last among sacrificable (keep LIVE WEB EVIDENCE)
    assert cb.SysBlock("diary", "d").sacrifice_rank == 0
    assert cb.SysBlock("mem", "m").sacrifice_rank == 3
    assert cb.SysBlock("pulse", "p").sacrifice_rank == 4
    assert cb.SysBlock("web", "w").sacrifice_rank == 6
