"""Last live-turn timestamp helpers (time-sense / activity; not an idle gate)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from src.config import settings

_log = logging.getLogger(__name__)


def _last_turn_at() -> datetime | None:
    try:
        from src.db.pool import get_pool

        with get_pool().connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT last_turn_at FROM agent_activity WHERE agent_id = %s",
                    (settings.default_agent_id,),
                )
                row = cur.fetchone()
        if row is None or row[0] is None:
            return None
        last = row[0]
        if last.tzinfo is None:
            return last.replace(tzinfo=timezone.utc)
        return last.astimezone(timezone.utc)
    except Exception as exc:
        _log.warning("agent_activity read failed (non-fatal): %s", exc)
        return None


def minutes_since_last_user_turn() -> float | None:
    """Minutes since last live graph turn, or None when no row."""
    last = _last_turn_at()
    if last is None:
        return None
    return (datetime.now(timezone.utc) - last).total_seconds() / 60.0


__all__ = [
    "minutes_since_last_user_turn",
]
