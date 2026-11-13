"""VisibleReply contract — channel-agnostic user-facing reply."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from langchain_core.messages import BaseMessage


@dataclass(frozen=True)
class VisibleReply:
    """What USER sees after model repair and think/main cut."""

    text: str
    thinking: str = ""
    is_fallback: bool = False
    tools: tuple[str, ...] = ()


def resolve_visible_reply(
    messages: Sequence[BaseMessage],
    *,
    user_message: str = "",
    preserve_thinking: bool = True,
    channel: str | None = None,
    tools: Sequence[str] = (),
) -> VisibleReply:
    """Single entry for live / API / background: visible text + optional think."""
    from src.agent_reply import resolve_agent_reply_text_with_meta
    from src.reply.visible import split_redacted_thinking
    from src.utils import get_message_text

    text, is_fallback = resolve_agent_reply_text_with_meta(
        messages,
        user_message=user_message,
        preserve_thinking=preserve_thinking,
        channel=channel,
    )
    thinking = ""
    if preserve_thinking and text and not is_fallback:
        think, _main, _trunc = split_redacted_thinking(text)
        if think.strip():
            thinking = think.strip()
        else:
            for msg in reversed(list(messages)):
                if (
                    getattr(msg, "type", None) == "ai"
                    or msg.__class__.__name__ == "AIMessage"
                ):
                    t, _m, _tr = split_redacted_thinking(get_message_text(msg))
                    if t.strip():
                        thinking = t.strip()
                    break
    return VisibleReply(
        text=text or "",
        thinking=thinking,
        is_fallback=bool(is_fallback),
        tools=tuple(tools or ()),
    )


__all__ = ["VisibleReply", "resolve_visible_reply"]
