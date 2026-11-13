"""Natural-language time parsing for Pulse intents (closed form list only)."""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import NamedTuple
from zoneinfo import ZoneInfo

from src.config import settings

_MIN_EVERY_MINUTES = 30

_ISO_FORMATS = (
    "%Y-%m-%dT%H:%M",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
)

_REL = re.compile(
    r"^in\s+(\d+)\s+(minute|minutes|hour|hours|day|days)\s*$",
    re.IGNORECASE,
)
# Bare unit: "in an hour", "in a minute".
_REL_BARE = re.compile(
    r"^in\s+(?:an?\s+)?(minute|hour|day)s?\s*$",
    re.IGNORECASE,
)
_TODAY = re.compile(
    r"^today\s+(?:at\s+)?(\d{1,2}):(\d{2})\s*$",
    re.IGNORECASE,
)
_TOMORROW = re.compile(
    r"^tomorrow\s+(?:at\s+)?(\d{1,2}):(\d{2})\s*$",
    re.IGNORECASE,
)
_DAILY = re.compile(
    r"^every\s+day\s+(\d{1,2}):(\d{2})\s*$",
    re.IGNORECASE,
)
_EVERY = re.compile(
    r"^every\s+(\d+)\s+(minute|minutes|hour|hours)\s*$",
    re.IGNORECASE,
)
_DAILY_RULE = re.compile(r"^daily\s+(\d{1,2}):(\d{2})$", re.IGNORECASE)
_EVERY_RULE = re.compile(r"^every\s+(\d+)([hm])$", re.IGNORECASE)

# Closed-list parts of day.
_PART_OF_DAY_ALT = r"morning|afternoon|evening|night"
_PART_OF_DAY_HM: dict[str, tuple[int, int]] = {
    "morning": (9, 0),
    "afternoon": (14, 0),
    "evening": (18, 0),
    "night": (22, 0),
}

# Weekday forms. Index = Monday=0 … Sunday=6.
_WEEKDAY_INDEX: dict[str, int] = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}
_WEEKDAY_ABBREV = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
_ABBREV_TO_INDEX = {name: i for i, name in enumerate(_WEEKDAY_ABBREV)}

_EN_WEEKDAY_ALT = "monday|tuesday|wednesday|thursday|friday|saturday|sunday"

# "on monday" [HH:MM|evening]; default 09:00
_WEEKDAY_ONCE = re.compile(
    rf"^on\s+({_EN_WEEKDAY_ALT})"
    rf"(?:\s+(\d{{1,2}}):(\d{{2}})|\s+({_PART_OF_DAY_ALT}))?\s*$",
    re.IGNORECASE,
)
# "every monday 09:00|evening"
_WEEKLY = re.compile(
    rf"^every\s+({_EN_WEEKDAY_ALT})"
    rf"(?:\s+(\d{{1,2}}):(\d{{2}})|\s+({_PART_OF_DAY_ALT}))\s*$",
    re.IGNORECASE,
)
_WEEKLY_RULE = re.compile(
    r"^weekly\s+(mon|tue|wed|thu|fri|sat|sun)\s+(\d{1,2}):(\d{2})$",
    re.IGNORECASE,
)

# "next wednesday" [HH:MM|evening]; default 09:00
_WEEKDAY_PLUS_WEEK = re.compile(
    rf"^next\s+({_EN_WEEKDAY_ALT})"
    rf"(?:\s+(\d{{1,2}}):(\d{{2}})|\s+({_PART_OF_DAY_ALT}))?\s*$",
    re.IGNORECASE,
)

# "around 9 in the evening" / "about 9pm" → today/tomorrow at that evening hour.
_EVENING_AROUND = re.compile(
    r"^(?:around\s+(\d{1,2})\s+in\s+the\s+evening|about\s+(\d{1,2})\s*(?:pm|in\s+the\s+evening))\s*$",
    re.IGNORECASE,
)
_TODAY_PART = re.compile(
    rf"^today\s+({_PART_OF_DAY_ALT})\s*$",
    re.IGNORECASE,
)
_TOMORROW_PART = re.compile(
    rf"^tomorrow\s+({_PART_OF_DAY_ALT})\s*$",
    re.IGNORECASE,
)

_WHEN_HINT = (
    "Examples: 'in 2 hours', 'in an hour', 'tomorrow at 09:00', 'on friday evening', "
    "'around 9 in the evening', 'next wednesday', '2026-07-09 21:00', "
    "'every day 21:00', 'every monday 09:00'. (crontab syntax is not supported.)"
)


class ParsedWhen(NamedTuple):
    run_at: datetime
    recur_rule: str | None


def _tz() -> ZoneInfo:
    return ZoneInfo(settings.agent_tz)


def _localize(now: datetime, tz: ZoneInfo) -> datetime:
    if now.tzinfo is None:
        return now.replace(tzinfo=tz)
    return now.astimezone(tz)


def _parse_hm(hour: str, minute: str) -> tuple[int, int]:
    h, m = int(hour), int(minute)
    if not (0 <= h <= 23 and 0 <= m <= 59):
        raise ValueError(f"Invalid time {h:02d}:{m:02d} — hour 0-23, minute 0-59.")
    return h, m


def _part_of_day_hm(word: str) -> tuple[int, int]:
    key = (word or "").casefold()
    if key in _PART_OF_DAY_HM:
        return _PART_OF_DAY_HM[key]
    raise ValueError(f"Unknown part of day: {word}")


def _clock_from_match(
    m: re.Match[str],
    *,
    hour_g: int,
    min_g: int,
    part_g: int,
    default: tuple[int, int] | None,
) -> tuple[int, int]:
    """HH:MM group, else part-of-day group, else default (or error if default is None)."""
    if m.group(hour_g) is not None:
        return _parse_hm(m.group(hour_g), m.group(min_g))
    if m.group(part_g) is not None:
        return _part_of_day_hm(m.group(part_g))
    if default is not None:
        return default
    raise ValueError(
        "Specify a time or a part of day (morning/afternoon/evening/night)."
    )


def _nearest_future(now: datetime, hour: int, minute: int) -> datetime:
    candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if candidate <= now:
        candidate += timedelta(days=1)
    return candidate


def _weekday_index(name: str) -> int:
    key = name.casefold()
    try:
        return _WEEKDAY_INDEX[key]
    except KeyError as exc:
        raise ValueError(f"Unknown weekday: {name}") from exc


def _next_weekday(now: datetime, weekday: int, hour: int, minute: int) -> datetime:
    """Next occurrence of weekday at hour:minute (strictly after ``now``)."""
    days_ahead = (weekday - now.weekday()) % 7
    candidate = (now + timedelta(days=days_ahead)).replace(
        hour=hour, minute=minute, second=0, microsecond=0
    )
    if candidate <= now:
        candidate += timedelta(days=7)
    return candidate


def _weekly_rule(weekday: int, hour: int, minute: int) -> str:
    return f"weekly {_WEEKDAY_ABBREV[weekday]} {hour:02d}:{minute:02d}"


def _minutes_for_every(n: int, unit: str) -> int:
    u = unit.casefold()
    if u.startswith("minute"):
        return n
    if u.startswith("hour"):
        return n * 60
    raise ValueError(f"Unknown interval unit: {unit}")


def _enforce_min_every(minutes: int) -> None:
    if minutes < _MIN_EVERY_MINUTES:
        raise ValueError(
            f"Minimum repeat interval is {_MIN_EVERY_MINUTES} minutes "
            f"(got {minutes})."
        )


def _parse_iso(raw: str, tz: ZoneInfo, now: datetime) -> datetime:
    text = raw.strip()
    for fmt in _ISO_FORMATS:
        try:
            dt = datetime.strptime(text, fmt)
        except ValueError:
            continue
        if fmt == "%Y-%m-%d":
            dt = dt.replace(hour=9, minute=0)
        dt = dt.replace(tzinfo=tz)
        if dt <= now:
            raise ValueError(
                f"Time {text} is already in the past — give a future date."
            )
        return dt
    raise ValueError(f"Unrecognised time form '{raw.strip()}'. {_WHEN_HINT}")


def parse_when(when: str, *, now: datetime) -> ParsedWhen:
    """Parse a closed-list when-string into (run_at, recur_rule).

    Timezone: ``settings.agent_tz``. Invalid input raises ``ValueError``
    with a human-readable message for tool feedback.
    """
    raw = (when or "").strip()
    if not raw:
        raise ValueError("Specify a time: when should the intent run.")

    tz = _tz()
    now = _localize(now, tz)
    lower = raw.casefold()

    # ISO (must be before generic checks)
    if re.match(r"^\d{4}-\d{2}-\d{2}", raw):
        return ParsedWhen(_parse_iso(raw, tz, now), None)

    m = _REL.match(lower)
    if m:
        n = int(m.group(1))
        unit = m.group(2).casefold()
        if unit.startswith("minute"):
            delta = timedelta(minutes=n)
        elif unit.startswith("hour"):
            delta = timedelta(hours=n)
        elif unit.startswith("day"):
            delta = timedelta(days=n)
        else:
            raise ValueError(f"Unknown unit: {unit}")
        return ParsedWhen(now + delta, None)

    m = _REL_BARE.match(lower)
    if m:
        unit = m.group(1).casefold()
        if unit.startswith("minute"):
            delta = timedelta(minutes=1)
        elif unit.startswith("hour"):
            delta = timedelta(hours=1)
        elif unit.startswith("day"):
            delta = timedelta(days=1)
        else:
            raise ValueError(f"Unknown unit: {unit}")
        return ParsedWhen(now + delta, None)

    m = _EVENING_AROUND.match(lower)
    if m:
        h = int(m.group(1) or m.group(2))
        if not (0 <= h <= 23):
            raise ValueError(f"Invalid hour {h} — hour 0-23.")
        # "around 9 in the evening" → 21:00; colloquial "around 12 at night" → midnight (0).
        if h == 12 or h == 0:
            h = 0
        elif h < 12:
            h += 12
        if h > 23:
            raise ValueError("Invalid hour after evening normalisation.")
        return ParsedWhen(_nearest_future(now, h, 0), None)

    m = _TODAY.match(lower)
    if m:
        h, mi = _parse_hm(m.group(1), m.group(2))
        return ParsedWhen(_nearest_future(now, h, mi), None)

    m = _TODAY_PART.match(lower)
    if m:
        h, mi = _part_of_day_hm(m.group(1))
        return ParsedWhen(_nearest_future(now, h, mi), None)

    m = _TOMORROW.match(lower)
    if m:
        h, mi = _parse_hm(m.group(1), m.group(2))
        base = now + timedelta(days=1)
        run_at = base.replace(hour=h, minute=mi, second=0, microsecond=0)
        return ParsedWhen(run_at, None)

    m = _TOMORROW_PART.match(lower)
    if m:
        h, mi = _part_of_day_hm(m.group(1))
        base = now + timedelta(days=1)
        run_at = base.replace(hour=h, minute=mi, second=0, microsecond=0)
        return ParsedWhen(run_at, None)

    m = _DAILY.match(lower)
    if m:
        h, mi = _parse_hm(m.group(1), m.group(2))
        rule = f"daily {h:02d}:{mi:02d}"
        return ParsedWhen(_nearest_future(now, h, mi), rule)

    # Weekly before bare weekday once — "every monday" must not hit the once form.
    m = _WEEKLY.match(lower)
    if m:
        wd = _weekday_index(m.group(1))
        h, mi = _clock_from_match(m, hour_g=2, min_g=3, part_g=4, default=None)
        rule = _weekly_rule(wd, h, mi)
        return ParsedWhen(_next_weekday(now, wd, h, mi), rule)

    m = _WEEKDAY_ONCE.match(lower)
    if m:
        wd = _weekday_index(m.group(1))
        h, mi = _clock_from_match(m, hour_g=2, min_g=3, part_g=4, default=(9, 0))
        return ParsedWhen(_next_weekday(now, wd, h, mi), None)

    m = _WEEKDAY_PLUS_WEEK.match(lower)
    if m:
        wd = _weekday_index(m.group(1))
        h, mi = _clock_from_match(m, hour_g=2, min_g=3, part_g=4, default=(9, 0))
        # Next matching weekday, then add one full week.
        base = _next_weekday(now, wd, h, mi)
        return ParsedWhen(base + timedelta(days=7), None)

    m = _EVERY.match(lower)
    if m:
        n = int(m.group(1))
        minutes = _minutes_for_every(n, m.group(2))
        _enforce_min_every(minutes)
        suffix = "m" if m.group(2).casefold().startswith("minute") else "h"
        if suffix == "m":
            rule = f"every {minutes}m"
            delta = timedelta(minutes=minutes)
        else:
            rule = f"every {n}h"
            delta = timedelta(hours=n)
        return ParsedWhen(now + delta, rule)

    raise ValueError(f"Unrecognised time form '{raw}'. {_WHEN_HINT}")


def next_run_at(recur_rule: str, after: datetime) -> datetime:
    """Next firing time for a recurring rule after ``after``."""
    rule = (recur_rule or "").strip()
    if not rule:
        raise ValueError("Empty repeat rule.")

    tz = _tz()
    after = _localize(after, tz)

    m = _DAILY_RULE.match(rule)
    if m:
        h, mi = _parse_hm(m.group(1), m.group(2))
        return _nearest_future(after, h, mi)

    m = _EVERY_RULE.match(rule)
    if m:
        n = int(m.group(1))
        unit = m.group(2).lower()
        minutes = n if unit == "m" else n * 60
        _enforce_min_every(minutes)
        if unit == "m":
            return after + timedelta(minutes=n)
        return after + timedelta(hours=n)

    m = _WEEKLY_RULE.match(rule)
    if m:
        wd = _ABBREV_TO_INDEX[m.group(1).lower()]
        h, mi = _parse_hm(m.group(2), m.group(3))
        return _next_weekday(after, wd, h, mi)

    raise ValueError(f"Unknown repeat rule: {recur_rule}")


__all__ = ["ParsedWhen", "next_run_at", "parse_when"]
