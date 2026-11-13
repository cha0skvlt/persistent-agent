"""Channel-agnostic VisibleReply helpers (think/main cut, outreach sanitize)."""

from src.reply.contract import VisibleReply, resolve_visible_reply
from src.reply.visible import (
    recover_think_main_split,
    sanitize_outreach_text,
    split_redacted_thinking,
    strip_leaked_tool_markup,
    strip_redacted_thinking,
    strip_think_tag_leaks,
)

__all__ = [
    "VisibleReply",
    "recover_think_main_split",
    "resolve_visible_reply",
    "sanitize_outreach_text",
    "split_redacted_thinking",
    "strip_leaked_tool_markup",
    "strip_redacted_thinking",
    "strip_think_tag_leaks",
]
