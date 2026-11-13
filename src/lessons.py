"""Causal lessons — structured behavioral change records (not diary prose)."""

from __future__ import annotations

import logging
import threading
from datetime import datetime
from time import perf_counter
from uuid import UUID

from src.config import settings
from src.db.pool import get_pool
from src.embed_status import mark_embed_unavailable
from src.memory import AgentMemory, _vector_param
from src.obs.logs import log_event

_log = logging.getLogger(__name__)
_default_lessons: CausalLessons | None = None
_lock = threading.Lock()


def get_default_lessons() -> CausalLessons:
    global _default_lessons
    if _default_lessons is None:
        with _lock:
            if _default_lessons is None:
                _default_lessons = CausalLessons()
    return _default_lessons


def reset_lessons_for_tests() -> None:
    global _default_lessons
    with _lock:
        _default_lessons = None


class CausalLessons:
    """Store / search ``change`` ← ``because`` rows for retrieve + tools."""

    def __init__(self, pool=None):
        self.pool = pool or get_pool()
        self._embed = AgentMemory(pool=self.pool)

    def save(
        self,
        change: str,
        because: str,
        *,
        evidence_ids: list[str] | None = None,
        agent: str | None = None,
        interactive: bool = False,
    ) -> tuple[str, bool] | None:
        """Persist a lesson. Returns ``(id, created)`` or ``None`` on failure."""
        change_t = (change or "").strip()
        because_t = (because or "").strip()
        if not change_t or not because_t:
            return None
        aid = settings.default_agent_id if agent is None else agent
        ids = [str(x).strip() for x in (evidence_ids or []) if str(x).strip()]
        # Exact dedupe before embed — avoid duplicate rows + curiosity ratchet.
        try:
            with self.pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT id::text
                        FROM causal_lessons
                        WHERE agent_id = %s
                          AND change_text = %s
                          AND because_text = %s
                        ORDER BY created_at DESC
                        LIMIT 1
                        """,
                        (aid, change_t, because_t),
                    )
                    existing = cur.fetchone()
            if existing and existing[0]:
                return str(existing[0]), False
        except Exception as exc:
            _log.warning("lesson dedupe lookup failed: %s", exc)
            # Fail closed — never INSERT when we cannot prove uniqueness.
            return None
        blob = f"{change_t} | {because_t}"
        try:
            vector = _vector_param(
                self._embed.get_vector(blob, interactive=interactive)
            )
        except Exception as exc:
            _log.warning("lesson embed failed: %s", exc)
            mark_embed_unavailable(str(exc))
            return None

        def _insert(*, use_on_conflict: bool) -> tuple[str, bool] | None:
            with self.pool.connection() as conn:
                with conn.cursor() as cur:
                    if use_on_conflict:
                        cur.execute(
                            """
                            INSERT INTO causal_lessons (
                              agent_id, change_text, because_text,
                              evidence_ids, embedding
                            )
                            VALUES (%s, %s, %s, %s, %s::vector)
                            ON CONFLICT (agent_id, change_text, because_text)
                            DO NOTHING
                            RETURNING id::text
                            """,
                            (aid, change_t, because_t, ids, vector),
                        )
                    else:
                        cur.execute(
                            """
                            INSERT INTO causal_lessons (
                              agent_id, change_text, because_text,
                              evidence_ids, embedding
                            )
                            VALUES (%s, %s, %s, %s, %s::vector)
                            RETURNING id::text
                            """,
                            (aid, change_t, because_t, ids, vector),
                        )
                    row = cur.fetchone()
                    if row is None and use_on_conflict:
                        cur.execute(
                            """
                            SELECT id::text
                            FROM causal_lessons
                            WHERE agent_id = %s
                              AND change_text = %s
                              AND because_text = %s
                            ORDER BY created_at DESC
                            LIMIT 1
                            """,
                            (aid, change_t, because_t),
                        )
                        existing = cur.fetchone()
                        conn.commit()
                        if existing and existing[0]:
                            return str(existing[0]), False
                        return None
                conn.commit()
            if not row or not row[0]:
                return None
            lid = str(row[0])
            log_event(
                "lesson.saved",
                lesson_id=lid,
                change_len=len(change_t),
                because_len=len(because_t),
                evidence_n=len(ids),
            )
            return lid, True

        try:
            return _insert(use_on_conflict=True)
        except Exception as exc:
            err = str(exc).lower()
            if "on conflict" in err and (
                "no unique" in err or "exclusion constraint" in err or "matching" in err
            ):
                _log.warning(
                    "lesson unique index missing; plain insert fallback: %s", exc
                )
                try:
                    return _insert(use_on_conflict=False)
                except Exception as exc2:
                    _log.warning("lesson save failed: %s", exc2)
                    return None
            _log.warning("lesson save failed: %s", exc)
            return None

    def search(
        self,
        query: str,
        *,
        limit: int = 3,
        agent: str | None = None,
        query_vector: list[float] | None = None,
        interactive: bool = False,
    ) -> list[tuple[str, str, str, float, datetime | None]]:
        """Return (id, change, because, score, created_at)."""
        aid = settings.default_agent_id if agent is None else agent
        lim = max(1, int(limit))
        t0 = perf_counter()
        try:
            if query_vector is not None:
                vector = _vector_param(query_vector)
            else:
                vector = _vector_param(
                    self._embed.get_vector(query, interactive=interactive)
                )
        except Exception as exc:
            _log.warning("lesson search embed failed: %s", exc)
            # A bad caller-supplied vector is not an embed-service outage.
            if query_vector is None:
                mark_embed_unavailable(str(exc))
            return []
        sql = """
            SELECT
              id::text,
              change_text,
              because_text,
              (1 - (embedding <=> %s::vector)) AS similarity,
              created_at
            FROM causal_lessons
            WHERE agent_id = %s
              AND embedding IS NOT NULL
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (vector, aid, vector, lim))
                rows = cur.fetchall()
        out: list[tuple[str, str, str, float, datetime | None]] = []
        for row in rows:
            created = row[4] if isinstance(row[4], datetime) else None
            out.append(
                (
                    str(row[0]),
                    str(row[1] or ""),
                    str(row[2] or ""),
                    float(row[3]),
                    created,
                )
            )
        log_event(
            "lesson.search",
            hits=len(out),
            latency_ms=round((perf_counter() - t0) * 1000.0, 2),
        )
        return out

    def latest(
        self, *, n: int = 3, agent: str | None = None
    ) -> list[tuple[str, str, str, datetime | None]]:
        aid = settings.default_agent_id if agent is None else agent
        lim = max(1, int(n))
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id::text, change_text, because_text, created_at
                    FROM causal_lessons
                    WHERE agent_id = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (aid, lim),
                )
                rows = cur.fetchall()
        out: list[tuple[str, str, str, datetime | None]] = []
        for row in rows:
            created = row[3] if isinstance(row[3], datetime) else None
            out.append((str(row[0]), str(row[1] or ""), str(row[2] or ""), created))
        return out


def format_lesson_line(
    lesson_id: str,
    change: str,
    because: str,
    created_at: datetime | None = None,
) -> str:
    date = created_at.strftime("%Y-%m-%d") if isinstance(created_at, datetime) else "?"
    return (
        f"[LESSON] [id={lesson_id}] ({date}) "
        f"changed: {change.strip()} — because: {because.strip()}"
    )


def parse_evidence_ids(raw: str | None) -> list[str]:
    text = (raw or "").strip()
    if not text:
        return []
    parts: list[str] = []
    for chunk in text.replace(";", ",").split(","):
        tok = chunk.strip()
        if not tok:
            continue
        try:
            parts.append(str(UUID(tok)))
        except ValueError:
            parts.append(tok)
    return parts


__all__ = [
    "CausalLessons",
    "format_lesson_line",
    "get_default_lessons",
    "parse_evidence_ids",
    "reset_lessons_for_tests",
]
