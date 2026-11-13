"""Agent tools for first-person diary (self_reflections lane)."""

from __future__ import annotations

from langchain_core.tools import tool

from src.config import settings
from src.diary_format import format_self_reflection_line
from src.memory import get_default_self_reflection
from src.memory_cleanup import parse_uuid_or_none


def _get_diary():
    return get_default_self_reflection()


@tool
def search_diary(query: str):
    """Search your first-person diary (self-reflections). Not world facts — use search_memory for those."""
    if not (query or "").strip():
        return []
    hits = _get_diary().search_with_ids(
        query,
        top_k=5,
        agent=settings.default_agent_id,
    )
    return [
        format_self_reflection_line(content, created_at, reflection_id=rid)
        for rid, content, _score, created_at in hits
        if (content or "").strip()
    ]


@tool
def diary_save(content: str):
    """Write a new first-person diary entry. For your own reflections, not world facts."""
    text = (content or "").strip()
    if not text:
        return "empty_content"
    if len(text) > 4000:
        text = text[:4000]
    try:
        from src.reverie.distill import gate_solitary_diary_write, requires_distill_gate

        if requires_distill_gate():
            rejected = gate_solitary_diary_write(text)
            if rejected:
                return rejected
    except Exception as exc:
        return f"distill_rejected: gate_failed ({type(exc).__name__})"
    saved = _get_diary().save(
        text,
        agent=settings.default_agent_id,
        source="tool",
    )
    return "saved" if saved else "unchanged (near-duplicate or blocked)"


@tool
def diary_replace(reflection_id: str, new_content: str):
    """Replace a diary entry by id. Use after search_diary when a reflection is wrong or noisy."""
    rid = (reflection_id or "").strip()
    if not parse_uuid_or_none(rid):
        return "not_found"
    text = (new_content or "").strip()
    if not text:
        return "empty_content"
    if len(text) > 4000:
        text = text[:4000]
    try:
        from src.reverie.distill import gate_solitary_diary_write, requires_distill_gate

        if requires_distill_gate():
            rejected = gate_solitary_diary_write(text)
            if rejected:
                return rejected
    except Exception as exc:
        return f"distill_rejected: gate_failed ({type(exc).__name__})"
    return _get_diary().replace(rid, text, agent=settings.default_agent_id)


@tool
def forget_diary(reflection_id: str):
    """Delete a diary entry by id. Seed rows (I am.) are protected."""
    rid = (reflection_id or "").strip()
    if not parse_uuid_or_none(rid):
        return "not_found"
    return _get_diary().delete(rid, agent=settings.default_agent_id)


__all__ = [
    "diary_replace",
    "diary_save",
    "forget_diary",
    "search_diary",
]
