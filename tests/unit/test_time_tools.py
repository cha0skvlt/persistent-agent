"""Unit tests for the temporal query tool helpers."""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from src.tools import time_tools as tt

TZ = ZoneInfo("UTC")
NOW = datetime(2026, 7, 7, 12, 0, 0, tzinfo=TZ)


def test_parse_when_relative_offsets():
    assert tt._parse_when("7d", tz=TZ, now=NOW) == NOW - timedelta(days=7)
    assert tt._parse_when("36h", tz=TZ, now=NOW) == NOW - timedelta(hours=36)
    assert tt._parse_when("2w", tz=TZ, now=NOW) == NOW - timedelta(weeks=2)
    assert tt._parse_when("1mo", tz=TZ, now=NOW) == NOW - timedelta(days=30)
    assert tt._parse_when("1y", tz=TZ, now=NOW) == NOW - timedelta(days=365)


def test_parse_when_iso_and_keywords():
    assert tt._parse_when("2026-05-01", tz=TZ) == datetime(2026, 5, 1, tzinfo=TZ)
    assert tt._parse_when("2026-05", tz=TZ) == datetime(2026, 5, 1, tzinfo=TZ)
    assert tt._parse_when("now", tz=TZ, now=NOW) == NOW
    assert tt._parse_when("", tz=TZ) is None
    assert tt._parse_when("garbage", tz=TZ) is None


def test_cadence_stats_basic():
    base = NOW
    ts = [
        base,
        base + timedelta(days=1),
        base + timedelta(days=1, hours=12),
        base + timedelta(days=6),
    ]
    stats = tt._cadence_stats(ts)
    assert stats is not None
    assert stats["count"] == 4
    assert stats["first"] == ts[0]
    assert stats["last"] == ts[-1]
    # gaps: 1d, 0.5d, 4.5d → max 4.5d ending at last ts
    assert round(stats["max_days"], 2) == 4.5
    assert stats["max_gap_end"] == ts[-1]


def test_cadence_stats_insufficient():
    assert tt._cadence_stats([]) is None
    assert tt._cadence_stats([NOW]) is None


def test_bucket_key_weekly_vs_monthly():
    dt = datetime(2026, 6, 15, 10, 0, tzinfo=TZ)
    assert tt._bucket_key(dt, TZ, weekly=False) == "2026-06"
    assert tt._bucket_key(dt, TZ, weekly=True).startswith("2026-W")


def test_dur_formatting():
    assert tt._dur(2.5) == "2.5d"
    assert tt._dur(0.5) == "12.0h"


def test_search_by_time_rejects_bad_mode_and_since(monkeypatch):
    monkeypatch.setattr(tt.settings, "agent_tz", "UTC", raising=False)
    out = tt.search_by_time.invoke({"mode": "nope"})
    assert any("unknown mode" in line for line in out)
    out2 = tt.search_by_time.invoke({"mode": "range", "since": "garbage"})
    assert any("unparseable since" in line for line in out2)
    out3 = tt.search_by_time.invoke({"mode": "range", "source": "vault"})
    assert any("unknown source" in line for line in out3)


def test_relative_until_alone_means_last_n(monkeypatch):
    monkeypatch.setattr(tt.settings, "agent_tz", "UTC", raising=False)
    captured: dict = {}

    def _fake_range(source, since, until, keyword, tz):
        captured["since"] = since
        captured["until"] = until
        return ["ok"]

    monkeypatch.setattr(tt, "_mode_range", _fake_range)
    out = tt.search_by_time.invoke({"mode": "range", "until": "7d"})
    assert out == ["ok"]
    assert captured["since"] is not None
    assert captured["until"] is not None
    assert captured["until"] > captured["since"]
