"""Novelty gate — embedding similarity vs recent diary entries.

Memory hygiene only: used by the diary echo gate to avoid near-duplicate
reflections. It does NOT gate proactive outreach (``message_user`` has no
dedup/cooldown). Fail-open when embeddings are unavailable (Ollama down).
"""

from __future__ import annotations

import logging
from typing import Literal

from src.config import settings

_log = logging.getLogger(__name__)

Lane = Literal["outreach", "diary"]

_DEFAULT_THRESHOLD = 0.88
_LOOKBACK = 10


def _threshold() -> float:
    val = float(settings.novelty_cosine_threshold)
    if val > 0:
        return val
    return _DEFAULT_THRESHOLD


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na <= 0 or nb <= 0:
        return 0.0
    return float(dot / (na * nb))


def _embed(text: str) -> list[float] | None:
    try:
        from src.memory import get_default_self_reflection

        vec = get_default_self_reflection().get_vector(text)
        if not vec:
            return None
        return list(vec)
    except Exception as exc:
        _log.warning("novelty embed failed: %s", exc)
        return None


def _recent_outreach_texts(agent_id: str, limit: int) -> list[str]:
    from src.db.pool import get_pool

    with get_pool().connection() as conn:
        rows = conn.execute(
            "SELECT draft_text FROM outbox WHERE agent_id = %s "
            "ORDER BY created_at DESC LIMIT %s",
            (agent_id, limit),
        ).fetchall()
    return [str(r[0]) for r in rows if r and r[0]]


def _recent_diary_texts(agent_id: str, limit: int) -> list[str]:
    from src.db.pool import get_pool

    with get_pool().connection() as conn:
        rows = conn.execute(
            "SELECT content FROM self_reflections WHERE agent_id = %s "
            "ORDER BY created_at DESC LIMIT %s",
            (agent_id, limit),
        ).fetchall()
    return [str(r[0]) for r in rows if r and r[0]]


def is_novel(
    text: str,
    *,
    lane: Lane = "outreach",
    agent_id: str | None = None,
    threshold: float | None = None,
) -> bool:
    """True when *text* is novel enough vs recent items in *lane*.

    Fail-open (returns True) when embedding is unavailable or lookback is empty.
    """
    raw = (text or "").strip()
    if not raw:
        return True
    thr = float(threshold if threshold is not None else _threshold())
    if thr <= 0:
        return True

    aid = agent_id or settings.default_agent_id
    try:
        if lane == "diary":
            recent = _recent_diary_texts(aid, _LOOKBACK)
        else:
            recent = _recent_outreach_texts(aid, _LOOKBACK)
    except Exception as exc:
        _log.warning("novelty lookback failed: %s", exc)
        return True

    if not recent:
        return True

    new_vec = _embed(raw)
    if new_vec is None:
        return True  # fail-open

    for draft in recent:
        old_vec = _embed(draft)
        if old_vec is None:
            continue
        if _cosine(new_vec, old_vec) >= thr:
            return False
    return True


__all__ = ["is_novel"]
