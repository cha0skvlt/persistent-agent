"""Update `current_time` in AgentState (default timezone from settings)."""

from __future__ import annotations

import logging
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from src.config import settings
from src.obs.logs import log_event
from src.obs.spans import traced_node
from src.state.agent_state import AgentState, checkpoint_overflow_updates

_log = logging.getLogger(__name__)


def _get_tz() -> ZoneInfo:
    return ZoneInfo(settings.agent_tz)


def _format_now(dt: datetime, tz_name: str) -> str:
    """Wall-clock string for SYS (no bound clock tool)."""
    return dt.strftime("%Y-%m-%d %H:%M:%S") + f" ({tz_name})"


def _next_turn_count(state: AgentState) -> int:
    return int(state.get("turn_count", 0) or 0) + 1


@traced_node("update_time")
def update_current_time(state: AgentState):
    """Graph entry: set wall-clock time for agent context."""
    log_event("node.start", node="update_time")
    t0 = time.perf_counter()
    try:
        msgs = list(state.get("messages") or [])
        sid = (state.get("session_uuid") or "").strip()
        anchor = (state.get("session_uuid_anchor") or "").strip()

        trim_ops = checkpoint_overflow_updates(msgs)

        tz_name = settings.agent_tz
        tz = _get_tz()
        now = datetime.now(tz)
        formatted = _format_now(now, tz_name)

        # New process: session_uuid no longer matches checkpoint anchor -> re-anchor the
        # summary window so session_len ignores pre-restart checkpoint tail.
        if sid and sid != anchor:
            offset = max(len(msgs), 0)
            turn = _next_turn_count(state)
            out: dict = {
                "current_time": formatted,
                "session_message_offset": offset,
                "session_uuid_anchor": sid,
                "session_uuid": sid,
                "turn_count": turn,
                "tool_rounds": 0,
                "turn_started_at": time.monotonic(),
                "tool_ledger": [],
            }
            if trim_ops:
                out["messages"] = trim_ops
            return out

        # Callers without session_uuid (tests): one-shot offset init only.
        if not sid:
            if "session_message_offset" not in state:
                offset = max(len(msgs), 0)
            else:
                offset = int(state.get("session_message_offset", 0) or 0)
            turn = _next_turn_count(state)
            out = {
                "current_time": formatted,
                "session_message_offset": offset,
                "turn_count": turn,
                "tool_rounds": 0,
                "turn_started_at": time.monotonic(),
                "tool_ledger": [],
            }
            if trim_ops:
                out["messages"] = trim_ops
            return out

        turn = _next_turn_count(state)
        out = {
            "current_time": formatted,
            "session_message_offset": int(state.get("session_message_offset", 0) or 0),
            "session_uuid": sid,
            "session_uuid_anchor": anchor,
            "turn_count": turn,
            "tool_rounds": 0,
            "turn_started_at": time.monotonic(),
            "tool_ledger": [],
        }
        if trim_ops:
            out["messages"] = trim_ops
        return out
    finally:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        log_event("node.timing", node="update_time", elapsed_ms=elapsed_ms)
