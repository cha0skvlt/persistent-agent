"""Heart metronome for Pulse intents — mark due, recover stale running."""

from __future__ import annotations

import logging

import psycopg

from src.config import settings
from src.obs.logs import log_event

_log = logging.getLogger(__name__)


def mark_due_intents(conn: psycopg.Connection) -> None:
    """Watchdog only: the API claim now picks pending rows by ``run_at`` directly;
    this marks due for visibility/metrics and recovers stuck running rows."""
    if not settings.pulse_enabled:
        return

    aid = settings.default_agent_id
    try:
        cur = conn.execute(
            """
            UPDATE intents SET status = 'due'
            WHERE agent_id = %s AND status = 'pending' AND run_at <= now()
            """,
            (aid,),
        )
        due_n = int(cur.rowcount or 0)
        if due_n > 0:
            log_event("heart.intents_due", count=due_n)

        ttl = int(settings.intents_running_ttl_min)
        # Park stale running as pending with a short backoff — not immediate due —
        # so recovery does not dump into a live-turn cancel/reclaim storm.
        cur = conn.execute(
            """
            UPDATE intents
            SET status = 'pending',
                run_at = now() + interval '2 minutes'
            WHERE agent_id = %s
              AND status = 'running'
              AND (
                    last_run_at < now() - make_interval(mins => %s)
                 OR last_run_at IS NULL
              )
            """,
            (aid, ttl),
        )
        stale_n = int(cur.rowcount or 0)
        if stale_n > 0:
            log_event("heart.intents_stale_recovered", count=stale_n)
    except Exception as exc:
        _log.warning('{"event":"pulse.error","detail":"%s"}', exc)


__all__ = ["mark_due_intents"]
