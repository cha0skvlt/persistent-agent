"""Unit tests for Pulse when_parser (closed-form when-list)."""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from src.config import settings
from src.pulse.when_parser import next_run_at, parse_when

TZ = ZoneInfo("UTC")
NOW = datetime(2026, 7, 8, 12, 0, 0, tzinfo=TZ)


@pytest.fixture(autouse=True)
def _utc_tz(monkeypatch):
    monkeypatch.setattr(settings, "agent_tz", "UTC", raising=False)


# --- ISO -----------------------------------------------------------------


def test_iso_datetime_t_separator():
    run_at, recur = parse_when("2026-07-09T21:00", now=NOW)
    assert recur is None
    assert run_at == datetime(2026, 7, 9, 21, 0, tzinfo=TZ)


def test_iso_datetime_space():
    run_at, recur = parse_when("2026-07-09 21:00", now=NOW)
    assert recur is None
    assert run_at == datetime(2026, 7, 9, 21, 0, tzinfo=TZ)


def test_iso_date_defaults_to_0900():
    run_at, _ = parse_when("2026-07-09", now=NOW)
    assert run_at == datetime(2026, 7, 9, 9, 0, tzinfo=TZ)


def test_iso_past_raises():
    with pytest.raises(ValueError, match="past"):
        parse_when("2026-07-07 10:00", now=NOW)


# --- Relative ------------------------------------------------------------


def test_relative_hours():
    run_at, recur = parse_when("in 2 hours", now=NOW)
    assert recur is None
    assert run_at == NOW + timedelta(hours=2)


def test_relative_minutes():
    run_at, _ = parse_when("in 45 minutes", now=NOW)
    assert run_at == NOW + timedelta(minutes=45)


def test_relative_days():
    run_at, _ = parse_when("in 3 days", now=NOW)
    assert run_at == NOW + timedelta(days=3)


def test_relative_bare_hour():
    run_at, recur = parse_when("in an hour", now=NOW)
    assert recur is None
    assert run_at == NOW + timedelta(hours=1)


def test_relative_bare_minute():
    run_at, _ = parse_when("in a minute", now=NOW)
    assert run_at == NOW + timedelta(minutes=1)


def test_tomorrow_with_optional_at():
    run_at, _ = parse_when("tomorrow at 10:00", now=NOW)
    assert run_at == datetime(2026, 7, 9, 10, 0, tzinfo=TZ)


def test_evening_around_nine():
    run_at, recur = parse_when("around 9 in the evening", now=NOW)
    assert recur is None
    assert run_at == datetime(2026, 7, 8, 21, 0, tzinfo=TZ)


def test_weekday_plus_week():
    # NOW=Wed 2026-07-08; next Wed is 07-15 09:00, +7 → 07-22
    run_at, recur = parse_when("next wednesday", now=NOW)
    assert recur is None
    assert run_at == datetime(2026, 7, 22, 9, 0, tzinfo=TZ)


def test_cron_still_rejected():
    with pytest.raises(ValueError, match="Unrecognised"):
        parse_when("0 9 * * *", now=NOW)


# --- Today / tomorrow ----------------------------------------------------


def test_today_future_same_day():
    run_at, _ = parse_when("today 18:00", now=NOW)
    assert run_at == datetime(2026, 7, 8, 18, 0, tzinfo=TZ)


def test_today_past_rolls_to_next_day():
    run_at, _ = parse_when("today 10:00", now=NOW)
    assert run_at == datetime(2026, 7, 9, 10, 0, tzinfo=TZ)


def test_tomorrow():
    run_at, _ = parse_when("tomorrow 09:00", now=NOW)
    assert run_at == datetime(2026, 7, 9, 9, 0, tzinfo=TZ)


# --- Daily recurring -----------------------------------------------------


def test_daily():
    run_at, recur = parse_when("every day 21:00", now=NOW)
    assert recur == "daily 21:00"
    assert run_at == datetime(2026, 7, 8, 21, 0, tzinfo=TZ)


def test_daily_past_time_today():
    run_at, recur = parse_when("every day 09:00", now=NOW)
    assert recur == "daily 09:00"
    assert run_at == datetime(2026, 7, 9, 9, 0, tzinfo=TZ)


def test_next_run_at_daily():
    after = datetime(2026, 7, 8, 22, 0, tzinfo=TZ)
    nxt = next_run_at("daily 21:00", after)
    assert nxt == datetime(2026, 7, 9, 21, 0, tzinfo=TZ)


def test_next_run_at_daily_same_day():
    after = datetime(2026, 7, 8, 20, 0, tzinfo=TZ)
    nxt = next_run_at("daily 21:00", after)
    assert nxt == datetime(2026, 7, 8, 21, 0, tzinfo=TZ)


# --- Every N hours/minutes -----------------------------------------------


def test_every_hours():
    run_at, recur = parse_when("every 2 hours", now=NOW)
    assert recur == "every 2h"
    assert run_at == NOW + timedelta(hours=2)


def test_every_minutes():
    run_at, recur = parse_when("every 45 minutes", now=NOW)
    assert recur == "every 45m"
    assert run_at == NOW + timedelta(minutes=45)


def test_every_one_hour():
    run_at, recur = parse_when("every 1 hour", now=NOW)
    assert recur == "every 1h"
    assert run_at == NOW + timedelta(hours=1)


def test_every_below_min_interval_raises():
    with pytest.raises(ValueError, match="30"):
        parse_when("every 15 minutes", now=NOW)
    with pytest.raises(ValueError, match="30"):
        parse_when("every 20 minutes", now=NOW)


def test_next_run_at_every_hours():
    after = datetime(2026, 7, 8, 12, 0, tzinfo=TZ)
    assert next_run_at("every 2h", after) == after + timedelta(hours=2)


def test_next_run_at_every_minutes():
    after = datetime(2026, 7, 8, 12, 0, tzinfo=TZ)
    assert next_run_at("every 30m", after) == after + timedelta(minutes=30)


# --- Weekday once / weekly -----------------------------------------------


def test_weekday_once_with_time():
    # NOW = Wed 2026-07-08 12:00 → next Friday
    run_at, recur = parse_when("on friday 09:00", now=NOW)
    assert recur is None
    assert run_at == datetime(2026, 7, 10, 9, 0, tzinfo=TZ)


def test_weekday_once_default_0900():
    run_at, recur = parse_when("on monday", now=NOW)
    assert recur is None
    assert run_at == datetime(2026, 7, 13, 9, 0, tzinfo=TZ)


def test_weekday_once_tuesday():
    run_at, _ = parse_when("on tuesday 18:00", now=NOW)
    assert run_at == datetime(2026, 7, 14, 18, 0, tzinfo=TZ)


def test_weekday_once_same_day_later():
    run_at, _ = parse_when("on wednesday 18:00", now=NOW)
    assert run_at == datetime(2026, 7, 8, 18, 0, tzinfo=TZ)


def test_weekday_once_same_day_past_rolls_week():
    run_at, _ = parse_when("on wednesday 09:00", now=NOW)
    assert run_at == datetime(2026, 7, 15, 9, 0, tzinfo=TZ)


def test_weekday_once_en():
    run_at, _ = parse_when("on monday 10:30", now=NOW)
    assert run_at == datetime(2026, 7, 13, 10, 30, tzinfo=TZ)


def test_weekly():
    run_at, recur = parse_when("every monday 09:00", now=NOW)
    assert recur == "weekly mon 09:00"
    assert run_at == datetime(2026, 7, 13, 9, 0, tzinfo=TZ)


def test_weekly_friday():
    run_at, recur = parse_when("every friday 21:00", now=NOW)
    assert recur == "weekly fri 21:00"
    assert run_at == datetime(2026, 7, 10, 21, 0, tzinfo=TZ)


def test_weekly_sunday():
    run_at, recur = parse_when("every sunday 08:00", now=NOW)
    assert recur == "weekly sun 08:00"
    assert run_at == datetime(2026, 7, 12, 8, 0, tzinfo=TZ)


def test_next_run_at_weekly():
    after = datetime(2026, 7, 13, 9, 0, tzinfo=TZ)  # Monday 09:00
    nxt = next_run_at("weekly mon 09:00", after)
    assert nxt == datetime(2026, 7, 20, 9, 0, tzinfo=TZ)


def test_next_run_at_weekly_later_same_week():
    after = datetime(2026, 7, 8, 12, 0, tzinfo=TZ)  # Wed
    nxt = next_run_at("weekly fri 09:00", after)
    assert nxt == datetime(2026, 7, 10, 9, 0, tzinfo=TZ)


# --- Errors --------------------------------------------------------------


def test_empty_raises():
    with pytest.raises(ValueError, match="Specify a time"):
        parse_when("", now=NOW)


def test_unknown_form_raises():
    with pytest.raises(ValueError, match="Unrecognised"):
        parse_when("on friday around eight", now=NOW)
    with pytest.raises(ValueError, match="Unrecognised"):
        parse_when("in a week", now=NOW)


def test_weekday_once_evening():
    run_at, recur = parse_when("on friday evening", now=NOW)
    assert recur is None
    assert run_at == datetime(2026, 7, 10, 18, 0, tzinfo=TZ)


def test_weekday_once_morning():
    run_at, _ = parse_when("on monday morning", now=NOW)
    assert run_at == datetime(2026, 7, 13, 9, 0, tzinfo=TZ)


def test_tomorrow_evening():
    run_at, recur = parse_when("tomorrow evening", now=NOW)
    assert recur is None
    assert run_at == datetime(2026, 7, 9, 18, 0, tzinfo=TZ)


def test_weekly_evening():
    run_at, recur = parse_when("every monday evening", now=NOW)
    assert recur == "weekly mon 18:00"
    assert run_at == datetime(2026, 7, 13, 18, 0, tzinfo=TZ)


def test_invalid_clock_raises():
    with pytest.raises(ValueError, match="Invalid time"):
        parse_when("today 25:00", now=NOW)


def test_next_run_at_unknown_rule():
    with pytest.raises(ValueError, match="Unknown repeat rule"):
        next_run_at("cron * * *", NOW)
