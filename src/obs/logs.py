"""Structured JSON log lines on the `agent.event` logger (unified with REPL pretty-printing)."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

_LOG = logging.getLogger("agent.event")


def _encode_event(event: str, fields: dict) -> str:
    payload = {"event": event, "ts_ms": int(time.time() * 1000), **fields}
    try:
        return json.dumps(payload, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        return json.dumps(
            {"event": event, "ts_ms": payload["ts_ms"], "error": "non_serializable"},
            ensure_ascii=False,
        )


def log_event(event: str, **fields: Any) -> None:
    """Emit one JSON object per line at INFO level. Visible in INFO and DEBUG modes."""
    if event == "node.timing":
        try:
            from src.obs.turn_timing import record_node_timing

            record_node_timing(
                str(fields.get("node") or ""),
                fields.get("elapsed_ms"),
            )
        except Exception:
            pass
    _LOG.info(_encode_event(event, fields))


def log_debug_event(event: str, **fields: Any) -> None:
    """Emit one JSON object per line at DEBUG level. Visible only in DEBUG mode."""
    if _LOG.isEnabledFor(logging.DEBUG):
        _LOG.debug(_encode_event(event, fields))


def is_debug_enabled() -> bool:
    """True when DEBUG mode is active — guards expensive string building before log calls."""
    return _LOG.isEnabledFor(logging.DEBUG)


def setup_agent_event_logger() -> None:
    """Attach `agent.event` to the root handler if not already set (idempotent)."""
    root = logging.getLogger()
    if not root.handlers:
        return
    _LOG.setLevel(logging.DEBUG)  # level gate handled by root; let root filter
    # Always propagate: events must reach the root file/console handler even when
    # an OTLP handler is attached to this logger — otherwise a downed collector
    # silently swallows all background-life events (reverie.*, outreach.*).
    _LOG.propagate = True


def format_elapsed_ms(ms: Any) -> str:
    """Human-readable duration for console logs.

    - < 500ms -> ``0s``
    - 500ms .. < 60000ms -> round to 0.5s steps, one decimal (e.g. ``1.5s``, ``32.0s``)
    - >= 60000ms -> ``Xm Ys`` (total seconds rounded from ms)
    """
    try:
        ms_f = float(ms)
    except (TypeError, ValueError):
        return str(ms)
    if ms_f < 0:
        ms_f = 0.0

    if ms_f >= 60_000:
        total_sec = int(round(ms_f / 1000.0))
        minutes, seconds = divmod(total_sec, 60)
        return f"{minutes}m {seconds}s"

    if ms_f < 500:
        return "0s"

    sec = ms_f / 1000.0
    half_steps = round(sec * 2.0) / 2.0
    if half_steps >= 60.0:
        total_sec = int(round(half_steps))
        minutes, seconds = divmod(total_sec, 60)
        return f"{minutes}m {seconds}s"
    return f"{half_steps:.1f}s"


def normalize_repl_user_text(s: str) -> str:
    """Single-line user text for the LLM: newlines from multiline paste -> spaces (REPL 8.6)."""
    t = (s or "").replace("\r\n", "\n").replace("\r", "\n")
    t = " ".join(part.strip() for part in t.split("\n"))
    return " ".join(t.split()).strip()
