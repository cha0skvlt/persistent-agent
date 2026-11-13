"""CRUD helpers for the ``vitals`` table.

All functions accept an open psycopg connection; callers manage transactions.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import psycopg

_log = logging.getLogger(__name__)


def fetch_latest_vitals_sync(
    conninfo: str | None = None,
    *,
    pool=None,
    connect_timeout: float = 3.0,
) -> dict[str, Any] | None:
    """Newest vitals row for diagnostics (indexed ``ORDER BY tick_at DESC``).

    When ``pool`` is provided, uses the app connection pool. Otherwise opens a
    one-off connection (heart daemon / tests).

    Returns ``None`` when the table is empty or Postgres is unreachable.
    """
    if pool is not None:
        with pool.connection() as conn:
            return latest_tick(conn)

    if conninfo is None:
        from src.config import settings

        conninfo = settings.database_url_plain_postgresql
    try:
        with psycopg.connect(conninfo, connect_timeout=int(connect_timeout)) as conn:
            return latest_tick(conn)
    except Exception as exc:
        _log.debug("vitals fetch_latest_vitals_sync skipped: %s", exc)
        return None


def write_tick(
    conn: "psycopg.Connection",
    *,
    pid: int,
    body_metrics: dict[str, Any],
    phylactery_status: str,
    signature: str | None = None,
) -> int | None:
    """Insert one vitals row. Returns tick_id on success, None on error.

    The heart connection uses ``autocommit=True``, so each statement is
    committed implicitly — no explicit commit/rollback needed.
    """
    try:
        row = conn.execute(
            """
            INSERT INTO vitals (heart_pid, body_metrics, phylactery_status, signature)
            VALUES (%s, %s::jsonb, %s, %s)
            RETURNING tick_id
            """,
            (pid, json.dumps(body_metrics), phylactery_status, signature),
        ).fetchone()
        return int(row[0]) if row else None
    except Exception as exc:
        _log.error("vitals write_tick failed: %s", exc)
        return None


def latest_tick(conn: "psycopg.Connection") -> dict[str, Any] | None:
    """Return the most recent vitals row as a dict, or None if table is empty."""
    try:
        row = conn.execute("""
            SELECT tick_id, tick_at, heart_pid, body_metrics,
                   phylactery_status, signature
            FROM vitals
            ORDER BY tick_at DESC
            LIMIT 1
            """).fetchone()
    except Exception as exc:
        _log.error("vitals latest_tick failed: %s", exc)
        return None
    if row is None:
        return None
    return {
        "tick_id": row[0],
        "tick_at": row[1],
        "heart_pid": row[2],
        "body_metrics": row[3],
        "phylactery_status": row[4],
        "signature": row[5],
    }


def tick_history(conn: "psycopg.Connection", limit: int = 20) -> list[dict[str, Any]]:
    """Return the last *limit* vitals rows, newest first."""
    try:
        rows = conn.execute(
            """
            SELECT tick_id, tick_at, heart_pid, body_metrics,
                   phylactery_status, signature
            FROM vitals
            ORDER BY tick_at DESC
            LIMIT %s
            """,
            (limit,),
        ).fetchall()
    except Exception as exc:
        _log.error("vitals tick_history failed: %s", exc)
        return []
    return [
        {
            "tick_id": r[0],
            "tick_at": r[1],
            "heart_pid": r[2],
            "body_metrics": r[3],
            "phylactery_status": r[4],
            "signature": r[5],
        }
        for r in rows
    ]
