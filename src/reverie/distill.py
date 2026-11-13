"""Solitary distill gate — durable writes are fact / lesson / hypothesis only."""

from __future__ import annotations

import re

# Raw think stays in logs; memory/diary get a short distill only.
_MAX_MEMORY_CHARS = 480
_MAX_DIARY_CHARS = 280
_MULTI_PARA_RE = re.compile(r"\n\s*\n")
_HYPOTHESIS_RE = re.compile(r"(?iu)\b(?:hypothesis|confidence|±|\+\-)")
_FIRST_PERSON_RUMINATE_RE = re.compile(r"(?iu)^\s*(?:i|me|my|mine)\b")


def is_distill_shaped(content: str, *, max_chars: int = _MAX_MEMORY_CHARS) -> bool:
    """True for a short fact / hypothesis±confidence — not an essay dump."""
    text = (content or "").strip()
    if not text or len(text) > max_chars:
        return False
    paras = [p for p in _MULTI_PARA_RE.split(text) if p.strip()]
    if len(paras) >= 3:
        return False
    if len(paras) >= 2 and len(text) > 240 and not _HYPOTHESIS_RE.search(text):
        return False
    # Long single-paragraph first-person rumination is essay sludge, not a fact.
    if (
        len(paras) <= 1
        and len(text) > 200
        and _FIRST_PERSON_RUMINATE_RE.search(text)
        and not _HYPOTHESIS_RE.search(text)
    ):
        return False
    return True


def requires_distill_gate() -> bool:
    """True on solitary/intent background passes — durable writes must be distill-shaped."""
    try:
        from src.reverie.turn_kind import is_background_turn

        return bool(is_background_turn())
    except Exception:
        return False


def _reject_ungrounded(reason: str) -> str:
    try:
        from src.obs.logs import log_event
        from src.obs.metrics import record_write_ungrounded_reject

        log_event("distill.rejected", reason=reason)
        record_write_ungrounded_reject(reason)
    except Exception:
        pass
    return "distill_rejected:no_evidence"


def _reject_invented_user() -> str:
    try:
        from src.obs.logs import log_event
        from src.obs.metrics import record_write_contamination

        log_event("distill.rejected", reason="invented_user_prefix")
        record_write_contamination("invented_user_prefix")
    except Exception:
        pass
    return "distill_rejected:no_evidence invented USER: prefix"


def gate_background_provenance(content: str, evidence: str | None = None) -> str | None:
    """Lexical/id reject-filter for durable writes. Heuristic, not provenance.

    This is a prefilter: it rejects writes with no token/id overlap against this
    turn's evidence. It does not prove the write follows from the source, and it
    must not be described as attribution. Pass ``evidence`` when the ContextVar
    may be empty (live extract, deferred).
    """
    text = (content or "").strip()
    if not text:
        return None
    from src.graph.turn_evidence import invented_user_prefix, write_has_turn_evidence

    if invented_user_prefix(text, evidence):
        return _reject_invented_user()
    if not write_has_turn_evidence(text, evidence):
        return _reject_ungrounded("no_overlap")
    return None


def gate_solitary_memory_write(content: str) -> str | None:
    """Return error string when background save_memory is not distill-shaped."""
    text = (content or "").strip()
    if not text:
        return None
    blocked = gate_background_provenance(text)
    if blocked:
        return blocked
    if is_distill_shaped(text, max_chars=_MAX_MEMORY_CHARS):
        return None
    return (
        "distill_rejected: background writes fact / lesson / hypothesis±confidence only "
        f"(≤{_MAX_MEMORY_CHARS} chars, no essay dump)."
    )


def gate_solitary_diary_write(content: str) -> str | None:
    """Return error string when background diary write is rejected."""
    text = (content or "").strip()
    if not text:
        return None
    blocked = gate_background_provenance(text)
    if blocked:
        return blocked
    if is_distill_shaped(text, max_chars=_MAX_DIARY_CHARS):
        return None
    return (
        "distill_rejected: background diary_replace accepts short distill "
        "only (no essay dump)."
    )


def distill_solitary_diary(text: str) -> str | None:
    """Accept short distill-shaped diary notes; reject essay dumps (no clip sludge)."""
    raw = (text or "").strip()
    if not raw:
        return None
    if gate_solitary_diary_write(raw):
        return None
    return raw


__all__ = [
    "distill_solitary_diary",
    "gate_background_provenance",
    "gate_solitary_diary_write",
    "gate_solitary_memory_write",
    "is_distill_shaped",
    "requires_distill_gate",
]
