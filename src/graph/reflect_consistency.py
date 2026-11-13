"""Deferred reflect write consistency — DB-only, same retrieval as tools.

Uses ``AgentMemory.search`` / ``SelfReflection.search_with_ids`` (embed + cosine),
not a separate search path. No LLM truth checks — compares against stored rows only.
"""

from __future__ import annotations

import logging
import re
from typing import Literal

from src.config import settings
from src.memory import AgentMemory, SelfReflection
from src.obs.logs import log_event

_log = logging.getLogger(__name__)

MemoryContradictionPolicy = Literal["log", "unverified", "replace"]
_VALID_POLICIES = frozenset({"log", "unverified", "replace"})

# High-similarity bar derived from MEMORY_SIMILARITY_FLOOR (retrieval drop floor).
_HIGH_SIM_RATIO = 0.8

_NEGATION_RE = re.compile(r"\b(not|never|no)\b", re.IGNORECASE)
_TOKEN_RE = re.compile(r"[^\W\d_]{3,}", re.UNICODE)


def high_similarity_threshold() -> float:
    """Similarity at or above this counts as a contradiction candidate."""
    raw = settings.reflect_consistency_gate_sim_threshold
    if raw is not None:
        return min(0.99, max(0.0, float(raw)))
    floor = max(0.0, min(1.0, float(settings.memory_similarity_floor)))
    return min(0.99, floor + (1.0 - floor) * _HIGH_SIM_RATIO)


def memory_contradiction_policy() -> MemoryContradictionPolicy:
    raw = (settings.memory_contradiction_policy or "log").strip().lower()
    if raw not in _VALID_POLICIES:
        _log.warning("Invalid MEMORY_CONTRADICTION_POLICY=%r; using log", raw)
        return "log"
    return raw  # type: ignore[return-value]


def texts_semantically_contradict(new_text: str, existing_text: str) -> bool:
    """Light polarity heuristic on overlapping tokens — not a truth oracle."""
    new_s = (new_text or "").strip()
    old_s = (existing_text or "").strip()
    if not new_s or not old_s:
        return False
    na = bool(_NEGATION_RE.search(new_s))
    nb = bool(_NEGATION_RE.search(old_s))
    if na == nb:
        return False
    new_tokens = {w.casefold() for w in _TOKEN_RE.findall(new_s)}
    old_tokens = {w.casefold() for w in _TOKEN_RE.findall(old_s)}
    overlap = len(new_tokens & old_tokens)
    min_overlap = 2 if (na or nb) else 3
    return overlap >= min_overlap


def find_memory_contradiction(
    mem: AgentMemory,
    new_text: str,
    *,
    top_k: int | None = None,
) -> tuple[str, str, float] | None:
    """Same retrieval path as hot retrieve (embed + cosine top-k over retrieve types)."""
    from src.graph.memory_distrust import parse_retrieve_memory_types

    k = max(1, int(top_k or settings.reflect_consistency_gate_top_k))
    types = parse_retrieve_memory_types(settings.memory_retrieve_types)
    try:
        hits = mem.search(
            new_text,
            limit=k,
            agent=settings.default_agent_id,
            memory_types=types,
            interactive=False,
        )
    except Exception as exc:
        _log.warning("reflect consistency memory search failed: %s", exc)
        return None
    if not hits:
        return None
    threshold = high_similarity_threshold()
    for hid, htxt, hscore in hits:
        if float(hscore) < threshold:
            continue
        if texts_semantically_contradict(new_text, str(htxt or "")):
            return str(hid), str(htxt or ""), float(hscore)
    return None


def find_diary_contradiction(
    diary: SelfReflection,
    new_text: str,
    *,
    top_k: int | None = None,
) -> tuple[str, str, float] | None:
    """Same retrieval path as ``search_diary`` (embed + cosine top-k)."""
    k = max(1, int(top_k or settings.reflect_consistency_gate_top_k))
    try:
        hits = diary.search_with_ids(
            new_text,
            top_k=k,
            agent=settings.default_agent_id,
            interactive=False,
        )
    except Exception as exc:
        _log.warning("reflect consistency diary search failed: %s", exc)
        return None
    if not hits:
        return None
    threshold = high_similarity_threshold()
    for rid, existing, score, _created in hits:
        if float(score) < threshold:
            continue
        if texts_semantically_contradict(new_text, str(existing or "")):
            return str(rid), str(existing or ""), float(score)
    return None


def log_contradiction_candidate(
    *,
    lane: str,
    new_text: str,
    existing_id: str,
    existing_text: str,
    similarity: float,
    policy: str,
) -> None:
    log_event(
        "reflect.contradiction_candidate",
        lane=lane,
        policy=policy,
        similarity=round(similarity, 4),
        similarity_threshold=round(high_similarity_threshold(), 4),
        existing_id=existing_id,
        new_preview=(new_text or "")[:120],
        existing_preview=(existing_text or "")[:120],
    )
    # ``log`` policy keeps both beliefs → pile-up signal for close-loop ops.
    # Only world-memory lane nudges retrieve knobs (diary polarity must not).
    lane_key = (lane or "").strip().lower()
    is_mem_lane = lane_key in {"memories", "memory"}
    resolution = (
        "log_pile"
        if (policy or "").strip().lower() == "log" and is_mem_lane
        else "seen"
    )
    try:
        from src.obs.metrics import record_memory_contradiction_event

        record_memory_contradiction_event(policy=policy, resolution=resolution)
    except Exception as exc:
        _log.warning("contradiction metrics failed: %s", exc)
    # Knob nudge must not depend on metrics health.
    if resolution == "log_pile":
        from src.policy_knobs import apply_outcome_safe

        apply_outcome_safe("log_pile")


def apply_memory_contradiction(
    mem: AgentMemory,
    new_text: str,
    conflict: tuple[str, str, float],
) -> bool:
    """Apply MEMORY_CONTRADICTION_POLICY for a world-fact write. Returns True if persisted."""
    existing_id, existing_text, score = conflict
    policy = memory_contradiction_policy()
    log_contradiction_candidate(
        lane="memories",
        new_text=new_text,
        existing_id=existing_id,
        existing_text=existing_text,
        similarity=score,
        policy=policy,
    )
    if policy == "replace":
        ok = mem.replace(existing_id, new_text, agent=settings.default_agent_id)
        if ok:
            log_event(
                "reflect.contradiction_resolved",
                lane="memories",
                action="replace",
                memory_id=existing_id,
            )
            try:
                from src.obs.metrics import record_memory_contradiction_event

                record_memory_contradiction_event(policy=policy, resolution="replace")
            except Exception as exc:
                _log.warning("contradiction replace metrics failed: %s", exc)
            return True
        log_event(
            "reflect.contradiction_skipped",
            lane="memories",
            action="replace_failed",
            memory_id=existing_id,
        )
        return False
    # Demote the conflicting prior row so retrieve distrust / gamma*importance
    # prefer fresher evidence over a contradicted belief.
    demote = float(settings.memory_contradiction_importance_factor)
    if 0.0 < demote < 1.0:
        if mem.scale_importance(existing_id, demote, agent=settings.default_agent_id):
            log_event(
                "reflect.contradiction_demoted",
                lane="memories",
                memory_id=existing_id,
                factor=demote,
            )
            # ``log`` already counted as log_pile in log_contradiction_candidate.
            if policy != "log":
                try:
                    from src.obs.metrics import record_memory_contradiction_event

                    record_memory_contradiction_event(
                        policy=policy, resolution="demoted"
                    )
                except Exception as exc:
                    _log.warning("contradiction demote metrics failed: %s", exc)
        else:
            _log.warning(
                "contradiction demote scale_importance failed id=%s factor=%s",
                existing_id,
                demote,
            )
    if policy == "unverified":
        result = mem.save(
            new_text,
            memory_type="unverified",
            agent=settings.default_agent_id,
        )
        if result is not None:
            try:
                from src.obs.metrics import record_memory_contradiction_event

                record_memory_contradiction_event(
                    policy=policy, resolution="unverified"
                )
            except Exception as exc:
                _log.warning("contradiction unverified metrics failed: %s", exc)
        return result is not None
    # log (default): save as normal semantic after logging the candidate.
    result = mem.save(
        new_text,
        memory_type="semantic",
        agent=settings.default_agent_id,
    )
    return result is not None


def apply_diary_contradiction(
    diary: SelfReflection,
    new_text: str,
    conflict: tuple[str, str, float],
    *,
    source: str = "reflect",
) -> bool:
    """Apply MEMORY_CONTRADICTION_POLICY for a diary write. Returns True if persisted."""
    existing_id, existing_text, score = conflict
    policy = memory_contradiction_policy()
    log_contradiction_candidate(
        lane="self_reflections",
        new_text=new_text,
        existing_id=existing_id,
        existing_text=existing_text,
        similarity=score,
        policy=policy,
    )
    if policy == "replace":
        status = diary.replace(existing_id, new_text, agent=settings.default_agent_id)
        if status == "replaced":
            log_event(
                "reflect.contradiction_resolved",
                lane="self_reflections",
                action="replace",
                reflection_id=existing_id,
            )
            return True
        log_event(
            "reflect.contradiction_skipped",
            lane="self_reflections",
            action="replace_failed",
            reflection_id=existing_id,
            status=status,
        )
        return False
    if policy == "unverified":
        log_event(
            "reflect.contradiction_skipped",
            lane="self_reflections",
            action="skip_unverified",
            reflection_id=existing_id,
        )
        try:
            from src.obs.metrics import record_memory_contradiction_event

            record_memory_contradiction_event(
                policy=policy, resolution="skip_unverified"
            )
        except Exception as exc:
            _log.warning("contradiction skip_unverified metrics failed: %s", exc)
        return False
    return diary.save(new_text, source=source)


def gate_memory_write(mem: AgentMemory, new_text: str) -> bool:
    """Deferred reflect consistency gate; return True when a write occurred."""
    conflict = find_memory_contradiction(mem, new_text)
    if conflict is None:
        result = mem.save(
            new_text,
            memory_type="semantic",
            agent=settings.default_agent_id,
        )
        return result is not None
    return apply_memory_contradiction(mem, new_text, conflict)


def gate_diary_write(
    diary: SelfReflection, new_text: str, *, source: str = "reflect"
) -> bool:
    """Deferred reflect consistency gate; return True when a write occurred."""
    conflict = find_diary_contradiction(diary, new_text)
    if conflict is None:
        return diary.save(new_text, source=source)
    return apply_diary_contradiction(diary, new_text, conflict, source=source)


__all__ = [
    "apply_diary_contradiction",
    "apply_memory_contradiction",
    "find_diary_contradiction",
    "find_memory_contradiction",
    "gate_diary_write",
    "gate_memory_write",
    "high_similarity_threshold",
    "memory_contradiction_policy",
    "texts_semantically_contradict",
]
