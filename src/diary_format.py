"""Shared formatting for self-reflection (diary) lines."""

from __future__ import annotations

from datetime import date, datetime


def self_reflection_date_label(created_at: datetime | date | None) -> str:
    if created_at is None:
        return "unknown"
    if isinstance(created_at, datetime):
        return created_at.strftime("%Y-%m-%d")
    return created_at.isoformat()


def format_self_reflection_line(
    content: str,
    created_at: datetime | date | None,
    *,
    reflection_id: str | None = None,
) -> str:
    """One diary line: — [id=…] (YYYY-MM-DD) text."""
    text = (content or "").strip()
    id_part = f"[id={reflection_id}] " if reflection_id else ""
    return f"— {id_part}({self_reflection_date_label(created_at)}) {text}"


__all__ = ["format_self_reflection_line", "self_reflection_date_label"]
