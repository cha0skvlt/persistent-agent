"""Unit tests for Pulse intent tools."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from src.config import settings
from src.tools.intent_tools import (
    cancel_intent_impl,
    list_intents_impl,
    schedule_intent_impl,
)

TZ = ZoneInfo("UTC")
NOW = datetime(2026, 7, 8, 12, 0, 0, tzinfo=TZ)


@pytest.fixture(autouse=True)
def _pulse_settings(monkeypatch):
    monkeypatch.setattr(settings, "agent_tz", "UTC", raising=False)
    monkeypatch.setattr(settings, "default_agent_id", "agent", raising=False)
    monkeypatch.setattr(settings, "pulse_enabled", True, raising=False)
    monkeypatch.setattr(settings, "intents_max_pending", 20, raising=False)
    monkeypatch.setattr(settings, "intents_max_per_day", 10, raising=False)


def test_schedule_solitary_rejects_day_plus_n_horizon(monkeypatch):
    monkeypatch.setattr(
        "src.reverie.turn_kind.is_solitary_turn",
        lambda: True,
    )
    out = schedule_intent_impl("Return to the topic", "in 3 days")
    assert "Not scheduled" in out
    assert "tomorrow" in out


def test_schedule_solitary_allows_tomorrow_evening_beyond_36h(monkeypatch):
    """Early morning → tomorrow 21:00 can exceed 36h wall-clock; still allowed."""
    monkeypatch.setattr(
        "src.reverie.turn_kind.is_solitary_turn",
        lambda: True,
    )
    early = datetime(2026, 7, 8, 8, 0, 0, tzinfo=TZ)

    def _fake_pool():
        conn = MagicMock()
        cur = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cur)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        cur.fetchone.side_effect = [(0,), (0,), ("id-tmr",)]
        pool = MagicMock()
        pool.connection.return_value.__enter__ = MagicMock(return_value=conn)
        pool.connection.return_value.__exit__ = MagicMock(return_value=False)
        return pool

    with (
        patch("src.db.pool.get_pool", _fake_pool),
        patch("src.mind.hub.record_intent_scheduled"),
        patch("src.tools.intent_tools.datetime") as mock_dt,
    ):
        mock_dt.now.return_value = early
        # parse_when uses real datetime; patch parse via when that resolves tomorrow 21:00
        out = schedule_intent_impl("return the thought", "tomorrow 21:00")
    assert out.startswith("Scheduled:")
    assert "2026-07-09 21:00" in out


def test_schedule_pulse_disabled(monkeypatch):
    monkeypatch.setattr(settings, "pulse_enabled", False, raising=False)
    assert (
        schedule_intent_impl("think", "in 1 hour") == "Not scheduled: pulse disabled."
    )


def test_schedule_invalid_when():
    assert "Unrecognised" in schedule_intent_impl("think", "someday later")


def test_schedule_weekday_when_parses(monkeypatch):
    """ "on friday" is valid when_parser input (needs DB mocks for insert)."""

    def _fake_pool():
        conn = MagicMock()
        cur = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cur)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        cur.fetchone.side_effect = [(0,), (0,), ("id-fri",)]
        pool = MagicMock()
        pool.connection.return_value.__enter__ = MagicMock(return_value=conn)
        pool.connection.return_value.__exit__ = MagicMock(return_value=False)
        return pool

    with (
        patch("src.db.pool.get_pool", _fake_pool),
        patch("src.mind.hub.record_intent_scheduled"),
        patch("src.tools.intent_tools.datetime") as mock_dt,
    ):
        mock_dt.now.return_value = NOW
        out = schedule_intent_impl("think", "on friday 09:00")
    assert out.startswith("Scheduled:")
    assert "2026-07-10 09:00" in out


def test_schedule_max_pending(monkeypatch):
    monkeypatch.setattr(settings, "intents_max_pending", 2, raising=False)

    def _fake_pool():
        conn = MagicMock()
        cur = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cur)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        cur.fetchone.side_effect = [(2,)]
        pool = MagicMock()
        pool.connection.return_value.__enter__ = MagicMock(return_value=conn)
        pool.connection.return_value.__exit__ = MagicMock(return_value=False)
        return pool

    with patch("src.db.pool.get_pool", _fake_pool):
        out = schedule_intent_impl("x", "in 1 hour")
    assert out.startswith("Not scheduled: too many pending intents")


def test_schedule_max_per_day(monkeypatch):
    monkeypatch.setattr(settings, "intents_max_per_day", 3, raising=False)
    executed: list[str] = []

    def _fake_pool():
        conn = MagicMock()
        cur = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cur)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        def _execute(sql, *args):
            executed.append(sql)

        cur.execute.side_effect = _execute
        cur.fetchone.side_effect = [(0,), (3,)]
        pool = MagicMock()
        pool.connection.return_value.__enter__ = MagicMock(return_value=conn)
        pool.connection.return_value.__exit__ = MagicMock(return_value=False)
        return pool

    with patch("src.db.pool.get_pool", _fake_pool):
        out = schedule_intent_impl("x", "in 1 hour")
    assert out == "Not scheduled: daily intent limit reached."
    daily_sql = next(s for s in executed if "24 hours" in s)
    assert "pending" in daily_sql and "due" in daily_sql and "running" in daily_sql


def test_schedule_success(monkeypatch):
    recorded: dict = {}

    def _fake_pool():
        conn = MagicMock()
        cur = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cur)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        cur.fetchone.side_effect = [(0,), (0,), ("abcd1234-uuid",)]
        pool = MagicMock()
        pool.connection.return_value.__enter__ = MagicMock(return_value=conn)
        pool.connection.return_value.__exit__ = MagicMock(return_value=False)
        return pool

    def _record(**kwargs):
        recorded.update(kwargs)

    with (
        patch("src.db.pool.get_pool", _fake_pool),
        patch("src.mind.hub.record_intent_scheduled", _record),
        patch("src.tools.intent_tools.datetime") as mock_dt,
    ):
        mock_dt.now.return_value = NOW
        out = schedule_intent_impl("think about X", "in 2 hours")

    assert out.startswith("Scheduled: think about X → 2026-07-08 14:00")
    assert recorded.get("text") == "think about X"
    assert recorded.get("run_at") == "2026-07-08 14:00"


def test_schedule_success_with_recur(monkeypatch):
    def _fake_pool():
        conn = MagicMock()
        cur = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cur)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        cur.fetchone.side_effect = [(0,), (0,), ("id-1",)]
        pool = MagicMock()
        pool.connection.return_value.__enter__ = MagicMock(return_value=conn)
        pool.connection.return_value.__exit__ = MagicMock(return_value=False)
        return pool

    with (
        patch("src.db.pool.get_pool", _fake_pool),
        patch("src.mind.hub.record_intent_scheduled"),
        patch("src.tools.intent_tools.datetime") as mock_dt,
    ):
        mock_dt.now.return_value = NOW
        out = schedule_intent_impl("evening", "every day 21:00")

    assert "(daily 21:00)" in out


def test_list_empty():
    def _fake_pool():
        conn = MagicMock()
        cur = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cur)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        cur.fetchall.side_effect = [[], []]
        pool = MagicMock()
        pool.connection.return_value.__enter__ = MagicMock(return_value=conn)
        pool.connection.return_value.__exit__ = MagicMock(return_value=False)
        return pool

    with patch("src.db.pool.get_pool", _fake_pool):
        assert list_intents_impl() == "No scheduled intents."


def test_list_formats_rows():
    run_at = datetime(2026, 7, 9, 9, 0, tzinfo=timezone.utc)

    def _fake_pool():
        conn = MagicMock()
        cur = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cur)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        cur.fetchall.side_effect = [
            [
                (
                    "abcdef12-0000-0000-0000-000000000001",
                    run_at,
                    "ask about X",
                    "pending",
                ),
            ],
            [("pending", 1), ("done", 2)],
        ]
        pool = MagicMock()
        pool.connection.return_value.__enter__ = MagicMock(return_value=conn)
        pool.connection.return_value.__exit__ = MagicMock(return_value=False)
        return pool

    with patch("src.db.pool.get_pool", _fake_pool):
        out = list_intents_impl()
    assert "abcdef12 — pending — 2026-07-09 09:00 — ask about X" in out
    assert "counts: pending=1 done/24h=2" in out


def test_cancel_not_found():
    def _fake_pool():
        conn = MagicMock()
        cur = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cur)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        cur.fetchall.return_value = []
        pool = MagicMock()
        pool.connection.return_value.__enter__ = MagicMock(return_value=conn)
        pool.connection.return_value.__exit__ = MagicMock(return_value=False)
        return pool

    with patch("src.db.pool.get_pool", _fake_pool):
        assert cancel_intent_impl("deadbeef") == "not_found"


def test_cancel_by_prefix():
    conn = MagicMock()
    cur = MagicMock()
    conn.cursor.return_value.__enter__ = MagicMock(return_value=cur)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    cur.fetchall.return_value = [("abcdef12-0000-0000-0000-000000000099",)]
    cur.rowcount = 1
    pool = MagicMock()
    pool.connection.return_value.__enter__ = MagicMock(return_value=conn)
    pool.connection.return_value.__exit__ = MagicMock(return_value=False)

    with patch("src.db.pool.get_pool", return_value=pool):
        assert cancel_intent_impl("abcdef12") == "cancelled"
    sql = " ".join(str(c.args[0]) for c in cur.execute.call_args_list)
    assert "last_run_at" in sql


def test_cancel_ambiguous_prefix():
    conn = MagicMock()
    cur = MagicMock()
    conn.cursor.return_value.__enter__ = MagicMock(return_value=cur)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    cur.fetchall.return_value = [
        ("abcdef12-0000-0000-0000-000000000001",),
        ("abcdef12-0000-0000-0000-000000000002",),
    ]
    pool = MagicMock()
    pool.connection.return_value.__enter__ = MagicMock(return_value=conn)
    pool.connection.return_value.__exit__ = MagicMock(return_value=False)

    with patch("src.db.pool.get_pool", return_value=pool):
        assert cancel_intent_impl("abcdef12") == "ambiguous_id"


def test_list_db_fail_returns_empty_message():
    def _fake_pool():
        conn = MagicMock()
        conn.cursor.return_value.__enter__.side_effect = RuntimeError("db down")
        pool = MagicMock()
        pool.connection.return_value.__enter__ = MagicMock(return_value=conn)
        pool.connection.return_value.__exit__ = MagicMock(return_value=False)
        return pool

    with patch("src.db.pool.get_pool", _fake_pool):
        assert list_intents_impl().startswith("list_failed:")


def test_cancel_running_returns_not_cancellable():
    def _fake_pool():
        conn = MagicMock()
        cur = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cur)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        cur.fetchall.return_value = [("abcdef12-0000-0000-0000-000000000099",)]
        cur.rowcount = 0
        pool = MagicMock()
        pool.connection.return_value.__enter__ = MagicMock(return_value=conn)
        pool.connection.return_value.__exit__ = MagicMock(return_value=False)
        return pool

    with patch("src.db.pool.get_pool", _fake_pool):
        assert cancel_intent_impl("abcdef12") == "not_cancellable"


@pytest.mark.unit
def test_intent_tools_bound_and_parity():
    from src.graph import workflow
    from src.prompt_tool_parity import assert_prompt_runtime_parity
    from src.selftest.checks import EXPECTED_AGENT_TOOLS

    names = {t.name for t in workflow.tools}
    for tool in ("schedule_intent", "list_intents", "cancel_intent"):
        assert tool in names
        assert tool in EXPECTED_AGENT_TOOLS
    assert_prompt_runtime_parity()


def test_schedule_roundtrip_db(disposable_postgres, monkeypatch):
    monkeypatch.setattr(settings, "pulse_enabled", True, raising=False)
    out = schedule_intent_impl("think about X", "in 2 hours")
    assert out.startswith("Scheduled: think about X →")

    from src.db.pool import get_pool

    with get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT status, text FROM intents WHERE agent_id = %s",
                (settings.default_agent_id,),
            )
            row = cur.fetchone()
    assert row is not None
    assert row[0] == "pending"
    assert row[1] == "think about X"
