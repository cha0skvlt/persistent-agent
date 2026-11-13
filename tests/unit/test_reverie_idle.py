"""Unit tests for last-turn activity helpers (time-sense)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import src.reverie.idle as idle_mod


def test_minutes_since_last_user_turn(monkeypatch):
    now = datetime.now(timezone.utc)
    monkeypatch.setattr(idle_mod, "_last_turn_at", lambda: now - timedelta(minutes=7))
    mins = idle_mod.minutes_since_last_user_turn()
    assert mins is not None
    assert 6.9 <= mins <= 7.1


def test_minutes_since_last_user_turn_none(monkeypatch):
    monkeypatch.setattr(idle_mod, "_last_turn_at", lambda: None)
    assert idle_mod.minutes_since_last_user_turn() is None
