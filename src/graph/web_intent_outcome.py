"""Live-turn web intent outcome — used / unused for policy knob nudges."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import ToolMessage

_WEB_TOOLS = frozenset({"search_web", "fetch_url", "browse_web"})


def _tool_body_usable(name: str, body: str) -> bool:
    """True when tool output is real evidence, not an error string."""
    text = (body or "").strip()
    if not text:
        return False
    low = text.lower()
    if name == "browse_web":
        if low.startswith("browse_error") or low.startswith("browse_closed"):
            return False
        if "disabled" in low and "browse" in low:
            return False
        return low.startswith("url:") or "---" in text
    if name == "search_web":
        from src.tools.web_evidence import web_search_body_usable

        return web_search_body_usable(text)
    if name == "fetch_url":
        from src.tools.web_evidence import web_fetch_body_usable

        return web_fetch_body_usable(text)
    return False


def turn_used_web(
    turn_msgs: list[Any],
    *,
    prefetch_usable: bool,
) -> bool:
    """True when this live turn already has usable web evidence (prefetch or tool)."""
    if prefetch_usable:
        return True
    for m in turn_msgs or []:
        if not isinstance(m, ToolMessage):
            continue
        name = str(getattr(m, "name", "") or "")
        if name not in _WEB_TOOLS:
            continue
        body = str(getattr(m, "content", "") or "")
        if _tool_body_usable(name, body):
            return True
    return False


def web_intent_outcome_signal(
    turn_msgs: list[Any],
    *,
    prefetch_usable: bool,
) -> str:
    """``web_intent_used`` or ``web_intent_unused``."""
    if turn_used_web(turn_msgs, prefetch_usable=prefetch_usable):
        return "web_intent_used"
    return "web_intent_unused"


__all__ = ["turn_used_web", "web_intent_outcome_signal"]
