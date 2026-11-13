"""Reply locale hints and fallback strings — English only."""

from __future__ import annotations

import re

ALLOWED_LANGUAGES = {"en"}

_INCOMPLETE_REPLY = "No visible model output — the turn ended without text."

LANGUAGE_SYS_RULE = (
    "Language (HARD): thinking and the visible reply are English only. "
    "Never think in another language and never answer in another one. "
    "Even if the user writes in another language, think and answer in English. "
    "Latin script only, except for names, tool names, code, URLs, and verbatim quotes. "
    "Never Cyrillic, CJK, Arabic, Thai, Hebrew, or Greek script."
)

_CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")


def turn_uses_cyrillic(text: str) -> bool:
    return bool(_CYRILLIC_RE.search(text or ""))


def incomplete_reply_message(user_text: str) -> str:
    """Locale string only — callers must use ``format_empty_turn_fallback``."""
    _ = user_text
    return _INCOMPLETE_REPLY
