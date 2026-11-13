"""User-visible agent/turn error formatting (Telegram gets debug detail)."""

from __future__ import annotations

import traceback
from typing import Sequence

from langchain_core.messages import BaseMessage, ToolMessage

from src.config import settings
from src.connect_errors import (
    is_transient_connection_error,
    ollama_unreachable_user_message,
)

_MAX_TELEGRAM_TRACE = 1200

_OPERATOR_ERROR_NEEDLES = (
    "turn error",
    "**turn error**",
    "operator notice:",
    "operator:",
    "no visible reply from the model",
    "no visible model output",
    "empty turn — check logs",
    "last tool results:",
    "internal error while invoking",
    "unable to generate parser",
    "no user query found in messages",
    "invalid_request_error",
    "invalidrequest_error",
    "graphrecursionerror",
    "raiseexception(",
)


def looks_like_operator_error(text: str) -> bool:
    """True for operator dumps / empty-turn notices — never agent speech or outreach."""
    t = (text or "").strip()
    if not t:
        return False
    low = t.casefold()
    return any(n in low for n in _OPERATOR_ERROR_NEEDLES)


def format_agent_turn_error(exc: BaseException, *, channel: str | None) -> str:
    """Format a failed turn for the user. Telegram includes exception type + traceback."""
    if is_transient_connection_error(exc):
        return ollama_unreachable_user_message(settings=settings, channel=channel)

    kind = type(exc).__name__
    msg = str(exc).strip() or "(no message)"
    ch = (channel or "").strip().lower()
    if ch == "telegram":
        tb = traceback.format_exc().strip()
        if len(tb) > _MAX_TELEGRAM_TRACE:
            tb = "…\n" + tb[-_MAX_TELEGRAM_TRACE:]
        # Operator notice — plain text (not agent speech; no markdown theater).
        return f"Turn error {kind}\n\n{msg}\n\n{tb}"
    return f"Internal error while invoking the agent. ({kind}: {msg})"


def format_empty_turn_fallback(
    messages: Sequence[BaseMessage],
    *,
    channel: str | None,
) -> str:
    """Operator incomplete notice when the graph produced no user-visible AI prose."""
    from src.text_locale import incomplete_reply_message
    from src.utils import last_real_human_index, last_real_human_text

    user_text = last_real_human_text(messages) or ""
    ch = (channel or "").strip().lower()
    last_human_idx = last_real_human_index(messages)
    tool_lines: list[str] = []
    scan = (
        messages[last_human_idx + 1 :] if last_human_idx is not None else list(messages)
    )
    for m in scan:
        if isinstance(m, ToolMessage):
            name = str(getattr(m, "name", None) or "tool")
            content = str(getattr(m, "content", "") or "").strip()
            if len(content) > 400:
                content = content[:400] + "…"
            tool_lines.append(f"{name}: {content or '(empty)'}")
        if len(tool_lines) >= 8:
            break
    base = incomplete_reply_message(user_text)
    body = f"Operator: {base}"
    if ch == "telegram":
        if tool_lines:
            return f"{body}\n\nLast tool results:\n" + "\n".join(tool_lines)
        return f"{body} (empty turn — check logs)"
    if tool_lines:
        return f"{body} Tool results: " + "; ".join(tool_lines[:3])
    return body


__all__ = [
    "format_agent_turn_error",
    "format_empty_turn_fallback",
    "looks_like_operator_error",
]
