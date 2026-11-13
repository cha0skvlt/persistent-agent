"""Solitary pacing gate — the only rhythm limit left.

Whether to write to USER is the agent's own decision (``message_user``); there
is no outreach cooldown, dedup, or idle gate. This module only paces how often
the background solitary reflection loop runs, from ``mind_events`` (last-run,
survives restart) + an in-process clock for same-process tests.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from src.config import settings


class RhythmAction(str, Enum):
    SOLITARY_PASS = "solitary_pass"


@dataclass(frozen=True)
class Verdict:
    allowed: bool
    reason: str = ""
    retry_after_sec: float = 0.0

    def __bool__(self) -> bool:
        return self.allowed


# In-process solitary clock — test helper / same-process fast path.
# Production interval prefers ``mind_events`` (survives API restart).
_last_solitary_at: float = 0.0


def reset_rhythm_state() -> None:
    """Test helper — clear in-process solitary clock."""
    global _last_solitary_at
    _last_solitary_at = 0.0


def mark_solitary_ran(at: float | None = None) -> None:
    global _last_solitary_at
    _last_solitary_at = float(at if at is not None else time.time())


def _solitary_interval_min() -> float:
    alias = settings.rhythm_solitary_interval_min
    if alias is not None:
        return float(alias)
    return float(settings.solitary_interval_min)


def _last_solitary_epoch() -> float:
    """Most recent solitary pass time: max(mind_events, in-process)."""
    mind_ts = 0.0
    try:
        from src.db.pool import get_pool

        with get_pool().connection() as conn:
            row = conn.execute(
                """
                SELECT created_at FROM mind_events
                WHERE agent_id = %s AND kind = 'solitary_pass'
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (settings.default_agent_id,),
            ).fetchone()
        if row and row[0] is not None:
            ts = row[0]
            if isinstance(ts, datetime):
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                mind_ts = ts.timestamp()
    except Exception:
        mind_ts = 0.0
    return max(mind_ts, _last_solitary_at)


def may(action: RhythmAction | str) -> Verdict:
    """Return whether *action* may proceed now."""
    name = action.value if isinstance(action, RhythmAction) else str(action)
    if name == RhythmAction.SOLITARY_PASS.value:
        return _may_solitary()
    return Verdict(False, reason="unknown_action")


def _may_solitary() -> Verdict:
    interval_sec = _solitary_interval_min() * 60.0
    # max(mind_events, in-process) — survives restart and same-process tests.
    last_at = _last_solitary_epoch()
    if last_at > 0 and interval_sec > 0:
        elapsed = time.time() - last_at
        if elapsed < interval_sec:
            return Verdict(
                False,
                reason="skip_interval",
                retry_after_sec=interval_sec - elapsed,
            )
    return Verdict(True)


__all__ = [
    "RhythmAction",
    "Verdict",
    "may",
    "mark_solitary_ran",
    "reset_rhythm_state",
]
