"""Deferred reflect diary echo gate — skip near-duplicate self-reflection rows."""

from __future__ import annotations

import logging
import math

from src.config import settings
from src.memory import SelfReflection, _is_protected_diary_content
from src.obs.logs import log_event

_log = logging.getLogger(__name__)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def is_diary_echo_exempt(content: str) -> bool:
    """Protected seed rows are outside echo checks."""
    return _is_protected_diary_content(content)


def find_diary_echo_match(
    diary: SelfReflection,
    new_text: str,
    *,
    check_n: int | None = None,
    threshold: float | None = None,
) -> tuple[str, str, float] | None:
    """Return (nearest_id, nearest_content, similarity) when echo exceeds threshold."""
    text = (new_text or "").strip()
    if not text or is_diary_echo_exempt(text):
        return None

    n = max(1, int(check_n if check_n is not None else settings.diary_echo_check_n))
    thr = float(threshold if threshold is not None else settings.diary_echo_threshold)

    try:
        new_vec = diary.get_vector(text, interactive=False)
    except Exception as exc:
        _log.warning("diary echo embed failed: %s", exc)
        return None

    recent = diary.recent_rows_with_embeddings(n=n, agent=settings.default_agent_id)
    if not recent:
        return None

    best_id = ""
    best_content = ""
    best_sim = -1.0
    for rid, content, emb in recent:
        if is_diary_echo_exempt(content):
            continue
        sim = _cosine_similarity(new_vec, emb)
        if sim > best_sim:
            best_id, best_content, best_sim = rid, content, sim

    if best_sim > thr:
        return best_id, best_content, best_sim
    return None


def skip_diary_echo_duplicate(diary: SelfReflection, new_text: str) -> bool:
    """True when the row should NOT be saved (echo duplicate). Logs skip event.

    Shared ``pulse.novelty.is_novel(lane=diary)`` can block duplicates; when it
    fail-opens or says novel, the local cosine lookback still applies.
    """
    text = (new_text or "").strip()
    if text and not is_diary_echo_exempt(text):
        try:
            from src.pulse.novelty import is_novel

            if not is_novel(text, lane="diary"):
                log_event(
                    "reflect.diary_echo_skipped",
                    nearest_id="",
                    similarity=None,
                    via="novelty",
                    preview=text[:60],
                )
                return True
        except Exception as exc:
            _log.warning("diary novelty check failed, falling back: %s", exc)

    match = find_diary_echo_match(diary, new_text)
    if match is None:
        return False
    nearest_id, nearest_content, similarity = match
    log_event(
        "reflect.diary_echo_skipped",
        nearest_id=nearest_id,
        similarity=round(similarity, 4),
        threshold=float(settings.diary_echo_threshold),
        new_preview=(new_text or "")[:120],
        nearest_preview=(nearest_content or "")[:120],
    )
    return True


__all__ = [
    "find_diary_echo_match",
    "is_diary_echo_exempt",
    "skip_diary_echo_duplicate",
]
