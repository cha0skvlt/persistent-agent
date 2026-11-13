"""Pulse intent tools — schedule, list, cancel agent intentions."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from langchain_core.tools import tool

from src.config import settings
from src.obs.logs import log_event
from src.pulse.when_parser import parse_when

_log = logging.getLogger(__name__)

_LIST_LIMIT = 10


def _tz() -> ZoneInfo:
    return ZoneInfo(settings.agent_tz)


def _fmt_run_at(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_tz())
    return dt.astimezone(_tz()).strftime("%Y-%m-%d %H:%M")


def schedule_intent_impl(text: str, when: str) -> str:
    body = (text or "").strip()
    if not body:
        return "Not scheduled: text is empty."
    if not settings.pulse_enabled:
        return "Not scheduled: pulse disabled."

    now = datetime.now(_tz())
    try:
        parsed = parse_when(when, now=now)
    except ValueError as exc:
        return str(exc)

    run_at, recur_rule = parsed.run_at, parsed.recur_rule
    aid = settings.default_agent_id

    from src.reverie.turn_kind import is_solitary_turn

    if is_solitary_turn() and not recur_rule:
        # Calendar through tomorrow (matches SYS), not rigid wall-clock 36h —
        # "tomorrow 21:00" from early morning can be >36h and is still allowed.
        if run_at.date() > (now.date() + timedelta(days=1)):
            return (
                "Not scheduled: solitary pass may only schedule through tomorrow "
                "(in N minutes/hours, today/tomorrow HH:MM) — not day+N revisits."
            )

    try:
        from src.db.pool import get_pool

        with get_pool().connection() as conn:
            with conn.cursor() as cur:
                # Serialize cap check + insert (pool may be autocommit).
                cur.execute(
                    "SELECT pg_advisory_lock(hashtext(%s))",
                    (f"agent:intent_cap:{aid}",),
                )
                try:
                    cur.execute(
                        "SELECT count(*) FROM intents WHERE agent_id = %s "
                        "AND status IN ('pending', 'due')",
                        (aid,),
                    )
                    pending = int((cur.fetchone() or [0])[0])
                    if pending >= int(settings.intents_max_pending):
                        return (
                            "Not scheduled: too many pending intents — "
                            "cancel or finish older ones."
                        )

                    cur.execute(
                        "SELECT count(*) FROM intents WHERE agent_id = %s "
                        "AND created_at > now() - interval '24 hours' "
                        "AND status IN ('pending', 'due', 'running')",
                        (aid,),
                    )
                    created_24h = int((cur.fetchone() or [0])[0])
                    if created_24h >= int(settings.intents_max_per_day):
                        return "Not scheduled: daily intent limit reached."

                    cur.execute(
                        """
                        INSERT INTO intents
                          (agent_id, text, run_at, recur_rule, status, created_by)
                        VALUES (%s, %s, %s, %s, 'pending', 'agent')
                        RETURNING id::text
                        """,
                        (aid, body, run_at, recur_rule),
                    )
                    intent_id = (cur.fetchone() or [None])[0]
                finally:
                    cur.execute(
                        "SELECT pg_advisory_unlock(hashtext(%s))",
                        (f"agent:intent_cap:{aid}",),
                    )
            conn.commit()
    except Exception as exc:
        _log.warning("schedule_intent failed: %s", exc)
        return "Not scheduled: database write failed."

    from src.mind.hub import record_intent_scheduled

    stamp = _fmt_run_at(run_at)
    record_intent_scheduled(text=body, run_at=stamp)
    log_event(
        "intent.scheduled",
        intent_id=intent_id,
        run_at=stamp,
        recur=recur_rule or "",
        preview=body[:60],
    )
    msg = f"Scheduled: {body} → {stamp}"
    if recur_rule:
        msg += f" ({recur_rule})"
    return msg


def list_intents_impl() -> str:
    aid = settings.default_agent_id
    try:
        from src.db.pool import get_pool

        with get_pool().connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id::text, run_at, text, status
                    FROM intents
                    WHERE agent_id = %s
                      AND status IN ('pending', 'due', 'running')
                    ORDER BY
                      CASE status
                        WHEN 'running' THEN 0
                        WHEN 'due' THEN 1
                        ELSE 2
                      END,
                      run_at ASC
                    LIMIT %s
                    """,
                    (aid, _LIST_LIMIT),
                )
                rows = cur.fetchall()
                cur.execute(
                    """
                    SELECT status, count(*)::int
                    FROM intents
                    WHERE agent_id = %s
                      AND (
                        status IN ('pending', 'due', 'running')
                        OR (
                          status IN ('done', 'cancelled', 'failed')
                          AND coalesce(last_run_at, created_at)
                              >= now() - interval '24 hours'
                        )
                      )
                    GROUP BY status
                    """,
                    (aid,),
                )
                counts = {str(s): int(n) for s, n in cur.fetchall()}
    except Exception as exc:
        _log.warning("list_intents failed: %s", exc)
        return f"list_failed: {str(exc)[:80]}"

    if not rows and not counts:
        return "No scheduled intents."

    lines: list[str] = []
    for intent_id, run_at, intent_text, status in rows:
        short = (intent_id or "")[:8]
        stamp = _fmt_run_at(run_at) if isinstance(run_at, datetime) else str(run_at)
        lines.append(f"{short} — {status} — {stamp} — {intent_text}")

    footer_bits = []
    for key in ("pending", "due", "running", "done", "cancelled", "failed"):
        n = counts.get(key)
        if n:
            label = key if key in ("pending", "due", "running") else f"{key}/24h"
            footer_bits.append(f"{label}={n}")
    if footer_bits:
        lines.append("counts: " + " ".join(footer_bits))

    if not rows:
        return "No open intents (pending/due/running).\n" + lines[-1]
    return "\n".join(lines)


def _resolve_intent_id(cur, intent_id: str) -> str | None:
    """Return full uuid text, ``None`` if missing, or ``ambiguous`` if prefix matches many."""
    raw = (intent_id or "").strip()
    if not raw:
        return None
    aid = settings.default_agent_id
    if len(raw) >= 32:
        cur.execute(
            "SELECT id::text FROM intents WHERE agent_id = %s AND id::text = %s LIMIT 1",
            (aid, raw),
        )
        row = cur.fetchone()
        return str(row[0]) if row and row[0] else None
    cur.execute(
        "SELECT id::text FROM intents WHERE agent_id = %s "
        "AND id::text LIKE %s ORDER BY id::text",
        (aid, raw + "%"),
    )
    rows = cur.fetchall() or []
    if not rows:
        return None
    if len(rows) > 1:
        return "ambiguous"
    return str(rows[0][0]) if rows[0] and rows[0][0] else None


def cancel_intent_impl(intent_id: str) -> str:
    try:
        from src.db.pool import get_pool

        with get_pool().connection() as conn:
            with conn.cursor() as cur:
                resolved = _resolve_intent_id(cur, intent_id)
                if resolved == "ambiguous":
                    return "ambiguous_id"
                if not resolved:
                    return "not_found"
                cur.execute(
                    """
                    UPDATE intents
                    SET status = 'cancelled', last_run_at = now()
                    WHERE id = %s::uuid AND status IN ('pending', 'due')
                    """,
                    (resolved,),
                )
                if (cur.rowcount or 0) < 1:
                    return "not_cancellable"
            conn.commit()
    except Exception as exc:
        _log.warning("cancel_intent failed: %s", exc)
        return f"cancel_failed: {str(exc)[:80]}"
    return "cancelled"


@tool
def schedule_intent(text: str, when: str) -> str:
    """Schedule your own future intention at a natural-language time.

    Copy USER's when literally when they gave one (e.g. "in 2 minutes").
    Examples: "in 2 minutes", "in 2 hours", "tomorrow evening",
    "on friday evening", "every day 21:00", "every monday 09:00".
    """
    return schedule_intent_impl(text, when)


@tool
def list_intents() -> str:
    """List your pending and due intentions (up to 10), ordered by run time."""
    return list_intents_impl()


@tool
def cancel_intent(intent_id: str) -> str:
    """Cancel a pending or due intention by full id or 8-character prefix."""
    return cancel_intent_impl(intent_id)


__all__ = [
    "cancel_intent",
    "cancel_intent_impl",
    "list_intents",
    "list_intents_impl",
    "schedule_intent",
    "schedule_intent_impl",
]
