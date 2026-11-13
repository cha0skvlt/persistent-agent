"""Hard reject rules for reflect-extracted facts before durable storage."""

from __future__ import annotations

import re

_GREETING_PREFIXES = (
    "good morning",
    "hello",
    "shalom",
    "hey",
    "hi",
)

_WORD_RE = re.compile(r"[A-Za-z]{4,}")


def is_junk(
    fact: str,
    *,
    user_text: str = "",
    agent_text: str = "",
) -> bool:
    """Return True when a reflect fact must not be persisted."""
    t = (fact or "").strip()
    if len(t) < 40:
        return True

    lower = t.casefold()
    for greeting in _GREETING_PREFIXES:
        if lower.startswith(greeting):
            return True

    if t.endswith("?") or t.endswith("!"):
        return True

    if not re.search(r"[A-Za-z]", t):
        return True

    if not _WORD_RE.search(t):
        return True

    norm = lower
    if user_text and norm == user_text.strip().casefold():
        return True
    if agent_text and norm == agent_text.strip().casefold():
        return True

    return False


__all__ = ["is_junk"]
