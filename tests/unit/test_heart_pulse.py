"""Unit tests for heart Pulse metronome."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.config import settings
from src.heart.pulse import mark_due_intents


@pytest.fixture(autouse=True)
def _pulse_heart_settings(monkeypatch):
    monkeypatch.setattr(settings, "pulse_enabled", True, raising=False)
    monkeypatch.setattr(settings, "default_agent_id", "agent", raising=False)
    monkeypatch.setattr(settings, "intents_running_ttl_min", 30, raising=False)


@pytest.mark.unit
def test_mark_due_intents_noop_when_pulse_disabled(monkeypatch):
    monkeypatch.setattr(settings, "pulse_enabled", False, raising=False)
    conn = MagicMock()
    mark_due_intents(conn)
    conn.execute.assert_not_called()


@pytest.mark.unit
def test_mark_due_intents_logs_due_count(monkeypatch):
    conn = MagicMock()
    conn.execute.side_effect = [MagicMock(rowcount=2), MagicMock(rowcount=0)]
    events: list[tuple[str, dict]] = []

    monkeypatch.setattr(
        "src.heart.pulse.log_event",
        lambda event, **fields: events.append((event, fields)),
    )
    mark_due_intents(conn)
    assert conn.execute.call_count == 2
    assert events == [("heart.intents_due", {"count": 2})]


@pytest.mark.unit
def test_mark_due_intents_logs_stale_recovery(monkeypatch):
    conn = MagicMock()
    conn.execute.side_effect = [MagicMock(rowcount=0), MagicMock(rowcount=1)]
    events: list[tuple[str, dict]] = []

    monkeypatch.setattr(
        "src.heart.pulse.log_event",
        lambda event, **fields: events.append((event, fields)),
    )
    mark_due_intents(conn)
    assert events == [("heart.intents_stale_recovered", {"count": 1})]


@pytest.mark.unit
def test_mark_due_intents_logs_both_counts(monkeypatch):
    conn = MagicMock()
    conn.execute.side_effect = [MagicMock(rowcount=3), MagicMock(rowcount=1)]
    events: list[str] = []

    monkeypatch.setattr(
        "src.heart.pulse.log_event",
        lambda event, **_: events.append(event),
    )
    mark_due_intents(conn)
    assert events == ["heart.intents_due", "heart.intents_stale_recovered"]


@pytest.mark.unit
def test_mark_due_intents_stale_sql_parks_pending_with_backoff(monkeypatch):
    """Stale running must not flip straight to due (cancel-thrash class)."""
    conn = MagicMock()
    conn.execute.side_effect = [MagicMock(rowcount=0), MagicMock(rowcount=1)]
    mark_due_intents(conn)
    stale_sql = conn.execute.call_args_list[1].args[0]
    assert "status = 'pending'" in stale_sql
    assert "interval '2 minutes'" in stale_sql
    assert "last_run_at IS NULL" in stale_sql
    assert "status = 'due'" not in stale_sql.split("UPDATE intents")[1]


@pytest.mark.unit
def test_mark_due_intents_swallows_db_error(monkeypatch, caplog):
    conn = MagicMock()
    conn.execute.side_effect = RuntimeError("db down")
    mark_due_intents(conn)
    assert "pulse.error" in caplog.text
