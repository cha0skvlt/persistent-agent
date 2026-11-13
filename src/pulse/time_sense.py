"""SYS time-sense block — idle delta and upcoming intents."""

from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from src.config import settings
from src.graph.context_budget import truncate_tail


def _tz() -> ZoneInfo:
    return ZoneInfo(settings.agent_tz)


def _fmt_run_at(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_tz())
    return dt.astimezone(_tz()).strftime("%Y-%m-%d %H:%M")


def format_human_delta(seconds: float) -> str:
    """Human-readable elapsed time: "3h 12m", "2d"."""
    total = max(0, int(seconds))
    days = total // 86400
    if days >= 1:
        return f"{days}d"
    hours = total // 3600
    mins = (total % 3600) // 60
    if hours > 0 and mins > 0:
        return f"{hours}h {mins}m"
    if hours > 0:
        return f"{hours}h"
    return f"{max(mins, 1)}m"


def _delta_line() -> str | None:
    from src.reverie.idle import minutes_since_last_user_turn

    elapsed = minutes_since_last_user_turn()
    if elapsed is None:
        return None
    delta = format_human_delta(elapsed * 60.0)
    return f"{delta} since the last USER message."


def _upcoming_intents_lines() -> list[str]:
    if not settings.pulse_enabled:
        return []
    try:
        from src.db.pool import get_pool

        with get_pool().connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT run_at, text FROM intents
                    WHERE agent_id = %s AND status IN ('pending', 'due')
                    ORDER BY run_at
                    LIMIT 5
                    """,
                    (settings.default_agent_id,),
                )
                rows = cur.fetchall()
    except Exception:
        return []
    if not rows:
        return []
    lines = ["Upcoming intents:"]
    for run_at, text in rows:
        if run_at is not None and run_at.tzinfo is None:
            run_at = run_at.replace(tzinfo=timezone.utc)
        when = _fmt_run_at(run_at) if run_at is not None else "?"
        lines.append(f"— {when} {text}")
    return lines


def time_sense_block() -> str | None:
    """Short SYS insert after Today is: idle delta + up to 5 upcoming intents."""
    parts: list[str] = []
    delta = _delta_line()
    if delta:
        parts.append(delta)
    intent_lines = _upcoming_intents_lines()
    if intent_lines:
        parts.extend(intent_lines)
    if not parts:
        return None
    body = "\n".join(parts)
    cap = int(settings.pulse_sys_max_chars)
    if cap > 0:
        body = truncate_tail(body, cap)
    return body.strip() or None


__all__ = ["format_human_delta", "time_sense_block"]
