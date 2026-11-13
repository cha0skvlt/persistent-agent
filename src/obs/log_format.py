"""Shared text clipping and key=value formatting for agent.event console output."""

from __future__ import annotations

import re
from typing import Any

LOG_PREVIEW_WORDS = 7

_WS_RE = re.compile(r"\s+")


def clip_one_line(text: str) -> str:
    """Collapse whitespace and newlines for log lines."""
    return _WS_RE.sub(" ", (text or "").strip())


def clip_words(text: str, *, max_words: int = LOG_PREVIEW_WORDS) -> str:
    """First *max_words* words; append … when truncated. Empty → ∅."""
    flat = clip_one_line(text)
    if not flat:
        return "∅"
    words = flat.split()
    if len(words) <= max_words:
        return flat
    return " ".join(words[:max_words]) + "…"


def fmt_preview(text: str, *, max_words: int = LOG_PREVIEW_WORDS) -> str:
    """Quoted preview for annotations (empty → ∅)."""
    s = clip_words(text, max_words=max_words)
    if s == "∅":
        return "∅"
    if '"' not in s:
        return f'"{s}"'
    return repr(s)


def fmt_size_chars(n: Any) -> str:
    """Compact character count (e.g. 12.4k)."""
    try:
        n_i = int(n)
    except (TypeError, ValueError):
        return str(n)
    if n_i >= 10_000:
        return f"{n_i / 1000:.0f}k"
    if n_i >= 1000:
        return f"{n_i / 1000:.1f}k"
    return f"{n_i}c"


def fmt_kv(*pairs: tuple[str, Any], sep: str = " ") -> str:
    """Space-separated key=value; skip None and empty strings."""
    parts: list[str] = []
    for key, val in pairs:
        if val is None or val == "":
            continue
        parts.append(f"{key}={val}")
    return sep.join(parts)


def fmt_score(score: Any) -> str:
    try:
        return f"{float(score):.3f}"
    except (TypeError, ValueError):
        return str(score)
