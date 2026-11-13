"""Temporal query tool — time as a dimension of the agent's own memory.

Point-in-time ("what now") comes from the ``update_time`` graph node (SYS),
not a bound clock tool. ``search_by_time`` is for *duration*: history along
``created_at`` across diary, world memory, and Mind Hub — ranges, cadence,
topic evolution.
"""

from __future__ import annotations

import logging
import re
import statistics
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from langchain_core.tools import tool

from src.config import settings

_log = logging.getLogger(__name__)

_MAX_ROWS = 40
_PREVIEW = 160

_REL_RE = re.compile(
    r"^(\d+(?:\.\d+)?)\s*(h|hr|hrs|hours?|d|days?|w|weeks?|mo|months?|y|years?)$",
    re.IGNORECASE,
)

_SOURCES = ("all", "diary", "memory", "timeline")


def _tz() -> ZoneInfo:
    return ZoneInfo(settings.agent_tz)


def _is_relative_when(value: str) -> bool:
    s = (value or "").strip().lower()
    return bool(_REL_RE.match(s))


def _parse_when(
    value: str, *, tz: ZoneInfo, now: datetime | None = None
) -> datetime | None:
    """Parse ISO date/datetime or a relative offset (`7d`, `36h`, `2w`, `1mo`)."""
    s = (value or "").strip().lower()
    if not s:
        return None
    now = now or datetime.now(tz)
    if s in ("now", "today"):
        return now

    m = _REL_RE.match(s)
    if m:
        n = float(m.group(1))
        unit = m.group(2).lower()
        if unit.startswith("mo"):
            return now - timedelta(days=30 * n)
        if unit.startswith("h"):
            return now - timedelta(hours=n)
        if unit.startswith("d"):
            return now - timedelta(days=n)
        if unit.startswith("w"):
            return now - timedelta(weeks=n)
        if unit.startswith("y"):
            return now - timedelta(days=365 * n)

    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d", "%Y-%m"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=tz)
        except ValueError:
            continue
    return None


def _fmt(dt: datetime, tz: ZoneInfo, *, with_time: bool = True) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo("UTC"))
    local = dt.astimezone(tz)
    return local.strftime("%Y-%m-%d %H:%M" if with_time else "%Y-%m-%d")


def _clip(text: str, limit: int = _PREVIEW) -> str:
    raw = " ".join((text or "").split())
    return raw if len(raw) <= limit else raw[: limit - 1] + "…"


def _dur(days: float) -> str:
    if days >= 1:
        return f"{days:.1f}d"
    hours = days * 24
    return f"{hours:.1f}h"


def _cadence_stats(ts_sorted: list[datetime]) -> dict | None:
    """Gap statistics for an ascending list of timestamps."""
    if len(ts_sorted) < 2:
        return None
    gaps = [
        (ts_sorted[i] - ts_sorted[i - 1]).total_seconds()
        for i in range(1, len(ts_sorted))
    ]
    max_gap = max(gaps)
    return {
        "count": len(ts_sorted),
        "first": ts_sorted[0],
        "last": ts_sorted[-1],
        "avg_days": statistics.mean(gaps) / 86400.0,
        "median_days": statistics.median(gaps) / 86400.0,
        "max_days": max_gap / 86400.0,
        "max_gap_end": ts_sorted[gaps.index(max_gap) + 1],
    }


def _bucket_key(dt: datetime, tz: ZoneInfo, *, weekly: bool) -> str:
    local = dt.astimezone(tz) if dt.tzinfo else dt.replace(tzinfo=tz)
    if weekly:
        iso = local.isocalendar()
        return f"{iso[0]}-W{iso[1]:02d}"
    return local.strftime("%Y-%m")


# --- DB access -----------------------------------------------------------


def _where_time(
    since: datetime | None, until: datetime | None, col: str = "created_at"
):
    clauses: list[str] = []
    params: list = []
    if since is not None:
        clauses.append(f"{col} >= %s")
        params.append(since)
    if until is not None:
        clauses.append(f"{col} <= %s")
        params.append(until)
    return clauses, params


def _fetch_rows(
    source: str,
    *,
    since: datetime | None,
    until: datetime | None,
    keyword: str | None,
    limit: int,
) -> list[tuple[datetime, str, str]]:
    """Return (created_at, label, content) rows across the chosen store(s)."""
    aid = settings.default_agent_id
    specs: list[tuple[str, str, str]] = []
    # (table, label_col, tag)
    if source in ("all", "diary"):
        specs.append(("self_reflections", "source", "diary"))
    if source in ("all", "memory"):
        specs.append(("memories", "memory_type", "mem"))
    if source in ("all", "timeline"):
        specs.append(("mind_events", "kind", "mind"))

    out: list[tuple[datetime, str, str]] = []
    from src.db.pool import get_pool

    per = max(5, limit // max(1, len(specs)))
    with get_pool().connection() as conn:
        for table, label_col, tag in specs:
            clauses, params = _where_time(since, until)
            sql_clauses = ["agent_id = %s"] + clauses
            sql_params: list = [aid] + params
            if keyword:
                sql_clauses.append("content ILIKE %s")
                sql_params.append(f"%{keyword}%")
            sql = (
                f"SELECT created_at, {label_col}, content FROM {table} "
                f"WHERE {' AND '.join(sql_clauses)} "
                f"ORDER BY created_at DESC LIMIT %s"
            )
            sql_params.append(per)
            with conn.cursor() as cur:
                cur.execute(sql, sql_params)
                for created_at, label, content in cur.fetchall():
                    if not content:
                        continue
                    lbl = f"{tag}:{label}" if label else tag
                    out.append((created_at, lbl, content))
    out.sort(key=lambda r: r[0])
    return out[-limit:]


def _fetch_timestamps(
    kind: str, *, since: datetime | None, until: datetime | None
) -> list[datetime]:
    aid = settings.default_agent_id
    clauses, params = _where_time(since, until)
    sql_clauses = ["agent_id = %s", "kind = %s"] + clauses
    sql_params: list = [aid, kind] + params
    sql = (
        "SELECT created_at FROM mind_events "
        f"WHERE {' AND '.join(sql_clauses)} ORDER BY created_at ASC"
    )
    from src.db.pool import get_pool

    with get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, sql_params)
            return [r[0] for r in cur.fetchall() if r and r[0]]


# --- modes ---------------------------------------------------------------


def _mode_range(source, since, until, keyword, tz) -> list[str]:
    rows = _fetch_rows(
        source, since=since, until=until, keyword=keyword, limit=_MAX_ROWS
    )
    if not rows:
        return ["Nothing in this range."]
    return [
        f"[{_fmt(created_at, tz)}] {label}: {_clip(content)}"
        for created_at, label, content in rows
    ]


def _cadence_line(name: str, stats: dict | None, tz) -> str:
    if not stats:
        return f"{name}: not enough data for a cadence."
    return (
        f"{name}: {stats['count']} events over "
        f"{_fmt(stats['first'], tz, with_time=False)}..{_fmt(stats['last'], tz, with_time=False)}; "
        f"typical gap {_dur(stats['median_days'])} "
        f"(mean {_dur(stats['avg_days'])}), "
        f"longest silence {_dur(stats['max_days'])} "
        f"until {_fmt(stats['max_gap_end'], tz, with_time=False)}."
    )


def _mode_cadence(since, until, tz) -> list[str]:
    user_ts = _fetch_timestamps("user_in", since=since, until=until)
    out_sent = _fetch_timestamps("outreach_sent", since=since, until=until)
    lines = [
        _cadence_line("USER writes", _cadence_stats(user_ts), tz),
        _cadence_line("I write first", _cadence_stats(out_sent), tz),
    ]
    return lines


def _mode_evolution(source, since, until, keyword, tz) -> list[str]:
    if not keyword:
        return ["evolution needs a query — the topic whose evolution to trace."]
    rows = _fetch_rows(source, since=since, until=until, keyword=keyword, limit=200)
    if not rows:
        return [f'"{keyword}" does not appear in this range.']
    span_days = (rows[-1][0] - rows[0][0]).total_seconds() / 86400.0
    weekly = span_days <= 60
    buckets: dict[str, list[tuple[datetime, str, str]]] = {}
    for row in rows:
        buckets.setdefault(_bucket_key(row[0], tz, weekly=weekly), []).append(row)

    lines = [
        f"Evolution of \"{keyword}\" ({len(rows)} mentions, step {'week' if weekly else 'month'}):"
    ]
    for key in sorted(buckets):
        chunk = buckets[key]
        sample = chunk[len(chunk) // 2]
        lines.append(f"[{key}] ×{len(chunk)}: {_clip(sample[2], 120)}")
    return lines


@tool
def search_by_time(
    mode: str = "range",
    query: str = "",
    since: str = "",
    until: str = "",
    source: str = "all",
):
    """Query own history along time — duration, not the current clock.

    mode: range | cadence | evolution.
    since/until: ISO ("2026-05-01") or relative ("7d", "36h", "2w"). Empty = open.
    source: all | diary | memory | timeline (ignored by cadence — always mind
    user_in / outreach_sent). Point-in-time "now" is already in SYS (update_time).
    """
    tz = _tz()
    m = (mode or "range").strip().lower()
    src = (source or "all").strip().lower()
    if m not in ("range", "cadence", "evolution"):
        return [f"unknown mode: {m!r} — use range|cadence|evolution"]
    if src not in _SOURCES:
        return [f"unknown source: {src!r} — use all|diary|memory|timeline"]
    kw = (query or "").strip() or None
    since_raw = (since or "").strip()
    until_raw = (until or "").strip()
    now = datetime.now(tz)
    dt_since = _parse_when(since_raw, tz=tz, now=now) if since_raw else None
    dt_until = _parse_when(until_raw, tz=tz, now=now) if until_raw else None
    if since_raw and dt_since is None:
        return [f"unparseable since: {since_raw!r}"]
    if until_raw and dt_until is None:
        return [f"unparseable until: {until_raw!r}"]
    # Bare relative until (no since) means "last N" → [now-N, now], not (-∞, now-N].
    if (
        until_raw
        and not since_raw
        and dt_until is not None
        and _is_relative_when(until_raw)
    ):
        dt_since, dt_until = dt_until, now

    try:
        if m == "cadence":
            return _mode_cadence(dt_since, dt_until, tz)
        if m == "evolution":
            return _mode_evolution(src, dt_since, dt_until, kw, tz)
        return _mode_range(src, dt_since, dt_until, kw, tz)
    except Exception as exc:  # noqa: BLE001
        _log.warning("search_by_time failed: %s", exc)
        return [f"search_by_time error: {str(exc)[:120]}"]


__all__ = ["search_by_time"]
