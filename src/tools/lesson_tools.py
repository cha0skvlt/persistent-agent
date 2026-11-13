"""Causal lesson tools — record structured behavioral change."""

from __future__ import annotations

from langchain_core.tools import tool

from src.lessons import get_default_lessons, parse_evidence_ids


def record_lesson_impl(
    change: str = "",
    because: str = "",
    evidence_ids: str = "",
) -> str:
    change_t = (change or "").strip()
    because_t = (because or "").strip()
    if not change_t or not because_t:
        return (
            "lesson_error: need both change= (what stopped/changed) "
            "and because= (why)."
        )
    try:
        from src.reverie.distill import is_distill_shaped, requires_distill_gate

        if requires_distill_gate():
            from src.reverie.distill import gate_background_provenance

            blocked = gate_background_provenance(f"{change_t} {because_t}")
            if blocked:
                return blocked
            # Background lessons stay short — no essay dump via record_lesson.
            if not is_distill_shaped(change_t, max_chars=480) or not is_distill_shaped(
                because_t, max_chars=480
            ):
                return (
                    "distill_rejected: background record_lesson needs short "
                    "change/because (≤480 chars each, no essay dump)."
                )
    except Exception as exc:
        return f"distill_rejected: gate_failed ({type(exc).__name__})"
    store = get_default_lessons()
    saved = store.save(
        change_t,
        because_t,
        evidence_ids=parse_evidence_ids(evidence_ids),
        interactive=True,
    )
    if not saved:
        return "lesson_error: save_failed (embed or DB)"
    lid, created = saved
    if created:
        from src.policy_knobs import apply_outcome_safe

        apply_outcome_safe("lesson_saved", lesson_id=lid)
        return f"lesson_saved id={lid}"
    return f"lesson_exists id={lid}"


def search_lessons_impl(query: str = "", limit: int = 3) -> str:
    q = (query or "").strip()
    if not q:
        return "lesson_error: empty query"
    store = get_default_lessons()
    hits = store.search(q, limit=max(1, min(8, int(limit or 3))), interactive=True)
    if not hits:
        return "lesson_empty: no matching lessons"
    lines = [
        f"[id={lid}] changed: {ch} — because: {bec} (score={score:.3f})"
        for lid, ch, bec, score, _created in hits
    ]
    return "\n".join(lines)


@tool
def record_lesson(change: str = "", because: str = "", evidence_ids: str = "") -> str:
    """Record a causal lesson: I changed/stopped X because Y (not a diary dump).

    Use when a real behavioral change is clear. Optional evidence_ids: comma-separated
    memory/diary ids that support the lesson.
    """
    return record_lesson_impl(change=change, because=because, evidence_ids=evidence_ids)


@tool
def search_lessons(query: str = "", limit: int = 3) -> str:
    """Search causal lessons (changed X because Y). Prefer this over diary cosine echo."""
    return search_lessons_impl(query=query, limit=limit)


__all__ = [
    "record_lesson",
    "record_lesson_impl",
    "search_lessons",
    "search_lessons_impl",
]
