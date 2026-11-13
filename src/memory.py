import json
import logging
import re
import threading
from collections.abc import Sequence
from datetime import datetime
from time import perf_counter
from typing import Any
from uuid import UUID
from zoneinfo import ZoneInfo

from src.config import settings
from src.db.pool import get_pool
from src.embed_status import mark_embed_unavailable
from src.obs.log_format import LOG_PREVIEW_WORDS, clip_words
from src.obs.logs import log_event

_DATE_PREFIX_RE = re.compile(r"^\[\d{4}-\d{2}-\d{2}\]")
# Injection-guard: reject attempts to plant fake system/control directives into durable memory.
# This is NOT content censorship — it protects memory integrity against prompt-injection attacks
# (e.g. a web-scraped snippet containing "# system ignore all previous instructions").
_MEMORY_INJECTION_GUARD_PATTERNS = [
    re.compile(r"(?im)^\s*#\s*system\b"),
    re.compile(r"(?im)\bignore\s+(all\s+)?previous\b"),
    re.compile(r"(?im)\byou\s+are\s+now\b"),
    re.compile(r"(?im)\bdeveloper\s+message\b"),
]


class _SkipDiaryDedup(Exception):
    """Internal: no embedding, so the cosine dedup query cannot run."""


_log = logging.getLogger(__name__)
_default_mem: "AgentMemory | None" = None
_singleton_lock = threading.Lock()


def _vector_param(vec) -> str:
    """Return pgvector literal string like '[1,2,3]' for %s::vector."""
    if not isinstance(vec, list) or not vec:
        raise ValueError(
            f"Invalid embedding vector: {type(vec)} len={len(vec) if hasattr(vec, '__len__') else 'n/a'}"
        )
    # pgvector accepts JSON-like array literal with brackets.
    return "[" + ",".join(str(float(x)) for x in vec) + "]"


def _memory_search_ann_limit(requested: int) -> int | None:
    """HNSW shortlist size; ``None`` selects legacy full-table rerank (``MEMORY_SEARCH_ANN_CANDIDATES=0``)."""
    cap = int(settings.memory_search_ann_candidates)
    if cap <= 0:
        return None
    return max(max(1, int(requested)), cap)


# Defaults for KG tool traversal (not used on retrieve_memory hot path).
_KG_TRAVERSE_POOL_SIZE = 30
_KG_TRAVERSE_MAX_EDGES = 10


def _normalize_memory_types(
    memory_type: str | None = None,
    memory_types: Sequence[str] | None = None,
) -> list[str]:
    if memory_types:
        out = [str(t).strip() for t in memory_types if str(t).strip()]
        if out:
            return out
    t = (memory_type or "semantic").strip() or "semantic"
    return [t]


def _memory_search_sql_and_params(
    *,
    vector: str,
    agent_id: str,
    memory_type: str = "semantic",
    memory_types: Sequence[str] | None = None,
    final_limit: int,
    return_embeddings: bool,
    return_rerank_fields: bool,
    ann_limit: int | None,
) -> tuple[str, tuple[Any, ...]]:
    final_limit = max(1, int(final_limit))
    types = _normalize_memory_types(memory_type, memory_types)

    if ann_limit is None:
        select_cols = "id, content, (1 - (embedding <=> %s::vector)) as similarity"
        if return_embeddings:
            select_cols += ", embedding, COALESCE(importance, 0.5) AS importance"
        elif return_rerank_fields:
            select_cols += ", COALESCE(importance, 0.5) AS importance"
        sql = f"""
            SELECT {select_cols}
            FROM memories
            WHERE agent_id = %s
              AND memory_type = ANY(%s)
              AND embedding IS NOT NULL
            ORDER BY (
              (1 - (embedding <=> %s::vector))
              * COALESCE(importance, 0.5)
            ) DESC
            LIMIT %s
            """
        return sql, (vector, agent_id, types, vector, final_limit)

    ann_limit = max(final_limit, int(ann_limit))
    if return_embeddings:
        outer_cols = "id, content, similarity, embedding, importance"
        inner_embedding = "\n                embedding,"
    elif return_rerank_fields:
        outer_cols = "id, content, similarity, importance"
        inner_embedding = ""
    else:
        outer_cols = "id, content, similarity"
        inner_embedding = ""
    sql = f"""
            SELECT {outer_cols}
            FROM (
              SELECT
                id,
                content,
                (1 - (embedding <=> %s::vector)) AS similarity,{inner_embedding}
                COALESCE(importance, 0.5) AS importance
              FROM memories
              WHERE agent_id = %s
                AND memory_type = ANY(%s)
                AND embedding IS NOT NULL
              ORDER BY embedding <=> %s::vector
              LIMIT %s
            ) AS ann
            ORDER BY (similarity * importance) DESC
            LIMIT %s
            """
    return sql, (vector, agent_id, types, vector, ann_limit, final_limit)


def _self_reflection_search_sql(
    *,
    vector: str,
    agent_id: str,
    limit: int,
    with_ids: bool,
) -> tuple[str, tuple[Any, ...]]:
    """Diary semantic search: HNSW ANN shortlist, then rerank by similarity*importance.

    ``MEMORY_SEARCH_ANN_CANDIDATES=0`` selects the legacy full-table scan
    (``ORDER BY score`` cannot use the vector index directly).
    """
    id_col = "id::text," if with_ids else ""
    ann_limit = _memory_search_ann_limit(limit)
    if ann_limit is None:
        sql = f"""
            SELECT
              {id_col}
              content,
              (1 - (embedding <=> %s::vector))
                * COALESCE(importance, 0.5) AS score,
              created_at
            FROM self_reflections
            WHERE agent_id = %s
              AND embedding IS NOT NULL
            ORDER BY score DESC
            LIMIT %s
            """
        return sql, (vector, agent_id, limit)
    sql = f"""
        SELECT {id_col} content, (similarity * importance) AS score, created_at
        FROM (
          SELECT
            {id_col}
            content,
            (1 - (embedding <=> %s::vector)) AS similarity,
            COALESCE(importance, 0.5) AS importance,
            created_at
          FROM self_reflections
          WHERE agent_id = %s
            AND embedding IS NOT NULL
          ORDER BY embedding <=> %s::vector
          LIMIT %s
        ) AS ann
        ORDER BY score DESC
        LIMIT %s
        """
    return sql, (vector, agent_id, vector, ann_limit, limit)


def _coerce_stored_embedding(val) -> list[float]:
    """Turn `memories.embedding` / pgvector driver value into plain floats for cosine math."""
    if val is None:
        return []
    if isinstance(val, list):
        return [float(x) for x in val]
    if isinstance(val, tuple):
        return [float(x) for x in val]
    if isinstance(val, str):
        s = val.strip()
        if s.startswith("[") and s.endswith("]"):
            inner = s[1:-1].strip()
            if not inner:
                return []
            return [float(x.strip()) for x in inner.split(",") if x.strip()]
    try:
        return [float(x) for x in val]
    except Exception:
        _log.warning("cannot coerce embedding column type=%s", type(val).__name__)
        return []


def sanitize_memory_fact(text: str) -> str:
    """Block durable-memory text that looks like prompt-injection (not content censorship)."""
    t = (text or "").strip()
    if not t:
        return ""
    for pat in _MEMORY_INJECTION_GUARD_PATTERNS:
        if pat.search(t):
            return ""
    return t


def _normalize_kg_part(value: str, *, max_len: int | None = None) -> str:
    """Normalize and sanitize one KG triple field."""
    text = sanitize_memory_fact(value)
    if max_len is not None:
        text = text[:max_len].strip()
    return text


def _normalize_kg_predicate(value: str) -> str:
    """Lowercase predicate with spaces → underscores (matches seed_kg whitelist)."""
    text = _normalize_kg_part(value, max_len=256)
    return re.sub(r"\s+", "_", text.lower())


def _normalize_kg_predicates_list(
    predicates: list[str] | None,
) -> list[str] | None:
    if not predicates:
        return None
    out = [
        p.strip().lower().replace(" ", "_")
        for p in predicates
        if isinstance(p, str) and p.strip()
    ]
    return out or None


def _memory_preview(content: str, *, words: int = LOG_PREVIEW_WORDS) -> str:
    """Return first words of saved content for compact console logging."""
    text = _DATE_PREFIX_RE.sub("", content or "", count=1).strip()
    return clip_words(text, max_words=words)


_TOKEN_RE = re.compile(r"\w{3,}", re.UNICODE)


def _is_text_near_duplicate(
    new: str, existing: str, threshold: float | None = None
) -> bool:
    """Lexical overlap fallback when cosine misses near-duplicates (new ⊆ existing ratio)."""
    overlap_thr = (
        float(threshold)
        if threshold is not None
        else float(settings.memory_dedup_text_overlap)
    )
    new_tokens = set(_TOKEN_RE.findall((new or "").lower()))
    if not new_tokens:
        return False
    existing_tokens = set(_TOKEN_RE.findall((existing or "").lower()))
    shared = len(new_tokens & existing_tokens)
    if not shared:
        return False
    ratio_new = shared / len(new_tokens)
    ratio_existing = shared / len(existing_tokens) if existing_tokens else 0.0
    return max(ratio_new, ratio_existing) >= overlap_thr


_DATE_PREFIX_RE = re.compile(r"^\s*\[[^\]]*\]\s*")
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")


def _fact_numbers(text: str) -> set[str]:
    """Numeric values in the claim, ignoring the ``[YYYY-MM-DD]`` stamp."""
    body = _DATE_PREFIX_RE.sub("", text or "")
    return {
        n.replace(",", ".").rstrip(".0").rstrip(".") for n in _NUMBER_RE.findall(body)
    }


def _numeric_conflict(new: str, existing: str) -> bool:
    """True when the two texts assert different numbers.

    A differing number is a differing claim, not a duplicate: "Everest 8848 m"
    and "Everest 9200 m" share every other token, so both cosine and token
    overlap rank them as near-identical and the second save was silently
    dropped as a dupe. That path quietly discards corrections — exactly the
    writes that matter most.
    """
    new_nums = _fact_numbers(new)
    old_nums = _fact_numbers(existing)
    if not new_nums or not old_nums:
        return False
    return new_nums != old_nums


def _memory_dedup_reason(
    content: str,
    existing_content: str,
    sim: float,
    *,
    cosine_thr: float,
    fuzzy_thr: float,
) -> str | None:
    if _numeric_conflict(content, existing_content):
        return None
    if sim >= cosine_thr:
        return "cosine"
    if sim >= fuzzy_thr and _is_text_near_duplicate(content, existing_content):
        return "fuzzy"
    return None


def get_default_memory() -> "AgentMemory":
    """Process-wide singleton AgentMemory (shared pool)."""
    global _default_mem
    if _default_mem is None:
        with _singleton_lock:
            if _default_mem is None:
                _default_mem = AgentMemory()
    return _default_mem


def reset_memory_singletons() -> None:
    """Reset process-wide memory singletons (use after close_pool in tests/e2e)."""
    global _default_mem, _default_self_reflection
    with _singleton_lock:
        _default_mem = None
        _default_self_reflection = None


class AgentMemory:
    def __init__(self, pool=None):
        self.model = settings.embed_model
        self.pool = pool or get_pool()

    def get_vector(self, text, *, interactive: bool = False):
        """Embed one string through the configured provider (Ollama or OpenAI-compatible)."""
        from src.embeddings import embed_text

        return embed_text(text, interactive=interactive)

    def save(
        self,
        content,
        memory_type: str = "semantic",
        agent: str | None = None,
        *,
        source: str = "reflect",
        source_ref: str | None = None,
    ) -> tuple[str, str] | None:
        """Persist a memory row. Returns ``(memory_id, status)`` or ``None`` if skipped.

        ``status`` is ``saved`` (new row) or ``dedup`` (near-duplicate skipped).
        ``source`` records where the write came from (``user``, ``tool``, ``web``,
        ``reflect``, ``background``) in ``metadata``. It is a label, not a proof
        of attribution.
        """
        aid = settings.default_agent_id if agent is None else agent
        t0 = perf_counter()
        content = sanitize_memory_fact(content)
        if not content:
            _log.info("Memory save skipped: injection pattern detected.")
            return None
        date_prefix = datetime.now(ZoneInfo(settings.agent_tz)).strftime("[%Y-%m-%d]")
        if not _DATE_PREFIX_RE.match(content):
            content = f"{date_prefix} {content}".strip()

        vector = _vector_param(self.get_vector(content))
        cosine_thr = float(settings.memory_dedup_cosine)
        fuzzy_thr = float(settings.memory_dedup_fuzzy_cosine)
        dedup_k = max(1, int(settings.memory_dedup_neighbors))
        with self.pool.connection() as conn:
            with conn.transaction():
                with conn.cursor() as cur:
                    # Serialize concurrent saves per (agent, type): dedup SELECT + INSERT
                    # have no unique constraint, so parallel saves could both pass dedup.
                    cur.execute(
                        "SELECT pg_advisory_xact_lock(hashtext(%s))",
                        (f"mem_save:{aid}:{memory_type}",),
                    )
                    try:
                        cur.execute(
                            """
                            SELECT
                              id,
                              content,
                              COALESCE(importance, %s) AS importance,
                              (1 - (embedding <=> %s::vector)) AS similarity
                            FROM memories
                            WHERE agent_id = %s
                              AND memory_type = %s
                            ORDER BY embedding <=> %s::vector
                            LIMIT %s
                            """,
                            (
                                settings.memory_dedup_base_importance,
                                vector,
                                aid,
                                memory_type,
                                vector,
                                dedup_k,
                            ),
                        )
                        for row in cur.fetchall():
                            if not row or row[3] is None:
                                continue
                            sim = float(row[3])
                            existing_content = str(row[1] or "")
                            reason = _memory_dedup_reason(
                                content,
                                existing_content,
                                sim,
                                cosine_thr=cosine_thr,
                                fuzzy_thr=fuzzy_thr,
                            )
                            if not reason:
                                continue
                            dedup_id = row[0]
                            _log.info(
                                "Memory dedup (%s): similarity=%.3f, id=%s, skip INSERT.",
                                reason,
                                sim,
                                dedup_id,
                            )
                            log_event(
                                "mem.dedup",
                                reason=reason,
                                old_id=str(dedup_id),
                                similarity=round(sim, 4),
                                preview=_memory_preview(content),
                                agent_id=aid,
                                latency_ms=round((perf_counter() - t0) * 1000.0, 2),
                            )
                            return str(dedup_id), "dedup"
                    except Exception as e:
                        _log.warning("Memory dedup failed; aborting save: %s", e)
                        raise

                    cur.execute(
                        """
                        INSERT INTO memories (
                          agent_id, memory_type, content, embedding, metadata,
                          source_ref
                        )
                        VALUES (%s, %s, %s, %s::vector, %s::jsonb, %s)
                        RETURNING id
                    """,
                        (
                            aid,
                            memory_type,
                            content,
                            vector,
                            json.dumps({"source": str(source or "reflect")}),
                            source_ref,
                        ),
                    )
                    row = cur.fetchone()
        if not row:
            return None
        mid = str(row[0])
        _log.info("Memory saved: %s...", content[:50])
        log_event(
            "mem.save",
            len=len(content),
            agent_id=aid,
            memory_id=mid,
            preview=_memory_preview(content),
            latency_ms=round((perf_counter() - t0) * 1000.0, 2),
        )
        return mid, "saved"

    def search(
        self,
        query,
        limit=3,
        agent: str | None = None,
        *,
        memory_type: str = "semantic",
        memory_types: Sequence[str] | None = None,
        return_embeddings: bool = False,
        return_rerank_fields: bool = False,
        query_vector: list[float] | None = None,
        interactive: bool = False,
    ):
        aid = settings.default_agent_id if agent is None else agent
        t0 = perf_counter()
        try:
            if query_vector is not None:
                vector = _vector_param(query_vector)
            else:
                vector = _vector_param(self.get_vector(query, interactive=interactive))
        except Exception as e:
            _log.warning(
                "Embeddings unavailable (model %s missing or Ollama down): %s - memory search skipped.",
                self.model,
                e,
            )
            mark_embed_unavailable(str(e))
            return []

        if return_embeddings and return_rerank_fields:
            raise ValueError(
                "return_embeddings and return_rerank_fields are mutually exclusive"
            )

        final_limit = max(1, int(limit))
        ann_limit = _memory_search_ann_limit(final_limit)
        sql, params = _memory_search_sql_and_params(
            vector=vector,
            agent_id=aid,
            memory_type=memory_type,
            memory_types=memory_types,
            final_limit=final_limit,
            return_embeddings=return_embeddings,
            return_rerank_fields=return_rerank_fields,
            ann_limit=ann_limit,
        )
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

        top_score = None
        if rows:
            try:
                top_score = float(rows[0][2])
            except Exception:  # pragma: no cover
                top_score = None
        log_event(
            "mem.search",
            query_len=len(query or ""),
            hits=len(rows),
            top_score=top_score,
            ann_candidates=ann_limit,
            latency_ms=round((perf_counter() - t0) * 1000.0, 2),
        )
        if return_embeddings:
            return [
                (
                    str(r[0]),
                    r[1],
                    float(r[2]),
                    _coerce_stored_embedding(r[3]),
                    float(r[4]),
                )
                for r in rows
            ]
        if return_rerank_fields:
            return [(str(r[0]), r[1], float(r[2]), float(r[3])) for r in rows]
        return [(str(r[0]), r[1], float(r[2])) for r in rows]

    def fetch_memory_embeddings_by_ids(
        self,
        memory_ids: list[str],
        *,
        agent: str | None = None,
    ) -> dict[str, list[float]]:
        """Load embedding vectors for rerank alignment (avoids wide SELECT in search)."""
        ids = [str(i).strip() for i in memory_ids if str(i).strip()]
        if not ids:
            return {}
        aid = settings.default_agent_id if agent is None else agent
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id::text, embedding
                    FROM memories
                    WHERE agent_id = %s
                      AND id = ANY(%s::uuid[])
                    """,
                    (aid, ids),
                )
                rows = cur.fetchall()
        return {
            str(row[0]): _coerce_stored_embedding(row[1])
            for row in rows
            if row and row[0]
        }

    def fetch_memory_types_by_ids(
        self,
        memory_ids: list[str],
        *,
        agent: str | None = None,
    ) -> dict[str, str]:
        """Load memory_type for rerank repeat-penalty skips (identity rows)."""
        meta = self.fetch_memory_rerank_meta_by_ids(memory_ids, agent=agent)
        return {mid: mtype for mid, (mtype, _created) in meta.items()}

    def fetch_memory_rerank_meta_by_ids(
        self,
        memory_ids: list[str],
        *,
        agent: str | None = None,
    ) -> dict[str, tuple[str, datetime | None]]:
        """Load memory_type + created_at for retrieve distrust / repeat-penalty skips."""
        ids = [str(i).strip() for i in memory_ids if str(i).strip()]
        if not ids:
            return {}
        aid = settings.default_agent_id if agent is None else agent
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id::text, memory_type, created_at
                    FROM memories
                    WHERE agent_id = %s
                      AND id = ANY(%s::uuid[])
                    """,
                    (aid, ids),
                )
                rows = cur.fetchall()
        out: dict[str, tuple[str, datetime | None]] = {}
        for row in rows:
            if not row or not row[0]:
                continue
            created = row[2] if isinstance(row[2], datetime) else None
            out[str(row[0])] = (str(row[1] or ""), created)
        return out

    def scale_importance(
        self,
        memory_id: str,
        factor: float,
        *,
        agent: str | None = None,
        floor: float = 0.05,
    ) -> bool:
        """Multiply importance by factor (clamped). Returns True when a row updated."""
        aid = settings.default_agent_id if agent is None else agent
        mid = (memory_id or "").strip()
        if not mid:
            return False
        try:
            mid = str(UUID(mid))
        except ValueError:
            return False
        mult = float(factor)
        if mult <= 0.0 or mult >= 1.0:
            return False
        flo = max(0.0, min(1.0, float(floor)))
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE memories
                    SET importance = GREATEST(
                      %s,
                      COALESCE(importance, 0.5) * %s
                    )
                    WHERE agent_id = %s
                      AND id = %s::uuid
                    """,
                    (flo, mult, aid, mid),
                )
                updated = cur.rowcount > 0
            conn.commit()
        return updated

    def replace(
        self,
        memory_id: str,
        new_content: str,
        *,
        agent: str | None = None,
    ) -> bool:
        """Replace one memory row by id. Returns True when updated."""
        aid = settings.default_agent_id if agent is None else agent
        mid = (memory_id or "").strip()
        if not mid:
            return False
        try:
            mid = str(UUID(mid))
        except ValueError:
            return False
        content = sanitize_memory_fact(new_content)
        if not content:
            _log.info("Memory replace skipped: injection pattern detected.")
            return False
        date_prefix = datetime.now(ZoneInfo(settings.agent_tz)).strftime("[%Y-%m-%d]")
        if not _DATE_PREFIX_RE.match(content):
            content = f"{date_prefix} {content}".strip()

        vector = _vector_param(self.get_vector(content))
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT memory_type, content
                    FROM memories
                    WHERE id = %s::uuid AND agent_id = %s
                    """,
                    (mid, aid),
                )
                row = cur.fetchone()
                if not row:
                    return False
                mem_type = str(row[0] or "")
                body = str(row[1] or "")
                if mem_type == "identity" or "[AFFIRMATION signed:" in body:
                    _log.info("Memory replace blocked: protected row.")
                    return False
                # Protection lives in the UPDATE itself: a row that became
                # protected between the SELECT and here is still not touched.
                cur.execute(
                    """
                    UPDATE memories
                    SET content = %s, embedding = %s::vector
                    WHERE id = %s::uuid AND agent_id = %s
                      AND memory_type <> 'identity'
                      AND content NOT LIKE '%%[AFFIRMATION signed:%%'
                    """,
                    (content, vector, mid, aid),
                )
                updated = cur.rowcount > 0
            conn.commit()
        if updated:
            log_event(
                "mem.replace",
                memory_id=mid,
                len=len(content),
                preview=_memory_preview(content),
            )
        return updated

    def save_conversation_snapshot(
        self,
        agent_id: str,
        messages: list[dict],
        summary: str | None = None,
        *,
        merge: bool = False,
    ):
        """Persist JSON dialogue into ``conversations`` (replace, or append when ``merge``)."""
        with self.pool.connection() as conn:
            with conn.transaction():
                with conn.cursor() as cur:
                    if merge:
                        merged: list[dict] = []
                        cur.execute(
                            """
                            SELECT messages FROM conversations
                            WHERE agent_id = %s
                            FOR UPDATE
                            """,
                            (agent_id,),
                        )
                        row = cur.fetchone()
                        prior = row[0] if row else None
                        if isinstance(prior, list):
                            merged = [dict(x) for x in prior if isinstance(x, dict)]
                        elif isinstance(prior, str) and prior.strip():
                            try:
                                parsed = json.loads(prior)
                                if isinstance(parsed, list):
                                    merged = [
                                        dict(x) for x in parsed if isinstance(x, dict)
                                    ]
                            except json.JSONDecodeError:
                                merged = []
                        merged.extend(messages)
                        cap = int(
                            getattr(settings, "conversation_snapshot_max_messages", 0)
                            or 0
                        )
                        if cap > 0 and len(merged) > cap:
                            merged = merged[-cap:]
                        payload = json.dumps(merged, ensure_ascii=False)
                    else:
                        payload = json.dumps(messages, ensure_ascii=False)
                    cur.execute(
                        """
                        INSERT INTO conversations (agent_id, messages, summary)
                        VALUES (%s, %s::jsonb, %s)
                        ON CONFLICT (agent_id) DO UPDATE SET
                          messages = EXCLUDED.messages,
                          summary = COALESCE(EXCLUDED.summary, conversations.summary)
                        """,
                        (agent_id, payload, summary),
                    )

    def update_conversation_summary(self, agent_id: str, summary: str) -> None:
        """Update rolling summary without changing the messages snapshot."""
        agent_id = (agent_id or "").strip() or "default"
        summary = (summary or "").strip()
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO conversations (agent_id, messages, summary)
                    VALUES (%s, '[]'::jsonb, %s)
                    ON CONFLICT (agent_id) DO UPDATE SET
                      summary = EXCLUDED.summary
                    """,
                    (agent_id, summary),
                )
            conn.commit()

    def save_fact_to_graph(
        self,
        subject: str,
        predicate: str,
        object_: str,
        agent: str | None = None,
    ) -> bool:
        from src.kg.entities import resolve_kg_entity, triple_hash_from_ids
        from src.kg.predicates import is_allowed

        aid = settings.default_agent_id if agent is None else agent
        subject = _normalize_kg_part(subject, max_len=256)
        predicate = _normalize_kg_predicate(predicate)
        object_ = _normalize_kg_part(object_)
        if not subject or not predicate or not object_:
            raise ValueError("subject/predicate/object must be non-empty")
        if not is_allowed(predicate):
            raise ValueError(f"predicate not allowed: {predicate!r}")

        text = f"{subject} {predicate} {object_}".strip()
        vector = _vector_param(self.get_vector(text))

        embed_cache: dict[str, list[float]] = {}
        with self.pool.connection() as conn:
            with conn.transaction():
                subject_id = resolve_kg_entity(
                    subject,
                    mem=self,
                    agent=aid,
                    conn=conn,
                    embed_cache=embed_cache,
                )
                object_id = resolve_kg_entity(
                    object_,
                    mem=self,
                    agent=aid,
                    conn=conn,
                    embed_cache=embed_cache,
                )
                triple_hash = triple_hash_from_ids(subject_id, predicate, object_id)

                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO knowledge_graph (
                          agent_id, subject, predicate, object,
                          subject_id, object_id,
                          triple_hash, embedding
                        )
                        VALUES (%s, %s, %s, %s, %s::uuid, %s::uuid, %s, %s::vector)
                        ON CONFLICT (agent_id, triple_hash) DO NOTHING
                        """,
                        (
                            aid,
                            subject,
                            predicate,
                            object_,
                            str(subject_id),
                            str(object_id),
                            triple_hash,
                            vector,
                        ),
                    )
                    if cur.rowcount == 0:
                        _log.info(
                            "Knowledge graph dedup: triple already exists, skipping insert."
                        )
                        return False
                    return True

    def search_graph(self, query: str, limit: int = 5, agent: str | None = None):
        aid = settings.default_agent_id if agent is None else agent
        try:
            vector = _vector_param(self.get_vector(query))
        except Exception as e:
            _log.warning(
                "Embeddings unavailable (model %s missing or Ollama down): %s - knowledge_graph search skipped.",
                self.model,
                e,
            )
            mark_embed_unavailable(str(e))
            return []

        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT subject, predicate, object, (1 - (embedding <=> %s::vector)) as similarity
                    FROM knowledge_graph
                    WHERE agent_id = %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (vector, aid, vector, max(1, int(limit))),
                )
                return cur.fetchall()

    def traverse_graph(
        self,
        start_entity_id: UUID,
        depth: int = 2,
        predicates: list[str] | None = None,
        agent: str | None = None,
        *,
        pool_size: int | None = None,
        max_edges: int | None = None,
    ) -> list[dict]:
        """
        Multi-hop walk on knowledge_graph from subject_id = start_entity_id.
        Returns edges {subject, predicate, object, hop}; depth capped at 3.

        Weighted random sample (Efraimidis-Spirakis) from a ranked pool
        so repeated retrieval does not always surface the same edges.
        """
        with self.pool.connection() as conn:
            return self._traverse_graph_on_conn(
                conn,
                start_entity_id,
                depth=depth,
                predicates=predicates,
                agent=agent,
                pool_size=pool_size,
                max_edges=max_edges,
            )

    def entities_matching_tokens(
        self, tokens: list[str], limit: int = 10, agent: str | None = None
    ) -> list[tuple[UUID, str]]:
        """Match entities by exact token/alias, bigrams, and pg_trgm similarity."""
        aid = settings.default_agent_id if agent is None else agent
        if not tokens or limit <= 0:
            return []
        raw = [t.strip().lower() for t in tokens if t.strip()]
        lowered = sorted({t for t in raw if len(t) >= 3})
        bigrams: list[str] = []
        for i in range(len(raw) - 1):
            a, b = raw[i], raw[i + 1]
            if len(a) >= 2 and len(b) >= 2:
                bigrams.append(f"{a} {b}")
        bigrams = sorted(set(bigrams))
        if not lowered and not bigrams:
            return []

        or_parts: list[str] = []
        params: list[Any] = [aid]
        if lowered:
            or_parts.append(
                "("
                "lower(trim(e.canonical)) = ANY(%s::text[]) "
                "OR (a.val IS NOT NULL AND lower(trim(a.val::text)) = ANY(%s::text[]))"
                ")"
            )
            params.extend([lowered, lowered])
            or_parts.append(
                "EXISTS (SELECT 1 FROM unnest(%s::text[]) AS ut(tok) "
                "WHERE length(tok) >= 3 AND lower(trim(e.canonical)) %% ut.tok)"
            )
            params.append(lowered)
            or_parts.append(
                "EXISTS (SELECT 1 FROM unnest(%s::text[]) AS ut(tok) "
                "WHERE length(tok) >= 3 AND a.val IS NOT NULL "
                "AND lower(trim(a.val::text)) %% ut.tok)"
            )
            params.append(lowered)
        if bigrams:
            or_parts.append(
                "("
                "lower(trim(e.canonical)) = ANY(%s::text[]) "
                "OR (a.val IS NOT NULL AND lower(trim(a.val::text)) = ANY(%s::text[]))"
                ")"
            )
            params.extend([bigrams, bigrams])

        where_or = " OR ".join(or_parts)
        sql = f"""
            SELECT DISTINCT ON (e.id) e.id::text, e.canonical
            FROM entities e
            LEFT JOIN LATERAL jsonb_array_elements_text(COALESCE(e.aliases, '[]'::jsonb)) AS a(val) ON true
            WHERE e.agent_id = %s
              AND ({where_or})
            ORDER BY e.id
            LIMIT %s
            """
        params.append(limit)

        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        out: list[tuple[UUID, str]] = []
        for row in rows:
            out.append((UUID(str(row[0])), str(row[1])))
        return out

    def traverse_graph_batch(
        self,
        start_entity_ids: list[UUID],
        depth: int = 2,
        predicates: list[str] | None = None,
        agent: str | None = None,
        *,
        pool_size: int | None = None,
        max_edges: int | None = None,
    ) -> dict[UUID, list[dict]]:
        """Run ``traverse_graph`` for several anchors on one pooled connection."""
        out: dict[UUID, list[dict]] = {}
        ids = list(start_entity_ids)
        if not ids:
            return out
        if max_edges is not None and int(max_edges) <= 0:
            return {eid: [] for eid in ids}
        n = len(ids)
        total_cap = (
            int(max_edges) if max_edges is not None else int(_KG_TRAVERSE_MAX_EDGES)
        )
        per_anchor = max(1, (total_cap + n - 1) // n)
        with self.pool.connection() as conn:
            for eid in ids:
                out[eid] = self._traverse_graph_on_conn(
                    conn,
                    eid,
                    depth=depth,
                    predicates=predicates,
                    agent=agent,
                    pool_size=pool_size,
                    max_edges=per_anchor,
                )
        return out

    def _traverse_graph_on_conn(
        self,
        conn,
        start_entity_id: UUID,
        *,
        depth: int,
        predicates: list[str] | None,
        agent: str | None,
        pool_size: int | None,
        max_edges: int | None,
    ) -> list[dict]:
        aid = settings.default_agent_id if agent is None else agent
        if max_edges is not None and int(max_edges) <= 0:
            return []
        max_hop = max(1, min(int(depth), 3))
        pool_size = max(1, int(pool_size or _KG_TRAVERSE_POOL_SIZE))
        if max_edges is None:
            edge_cap = max(1, int(_KG_TRAVERSE_MAX_EDGES))
        else:
            edge_cap = max(1, int(max_edges))
        params: list = [aid, str(start_entity_id)]
        anchor_pred = ""
        recursive_pred = ""
        predicates = _normalize_kg_predicates_list(predicates)
        if predicates:
            anchor_pred = " AND lower(kg.predicate) = ANY(%s::text[]) "
            recursive_pred = " AND lower(kg.predicate) = ANY(%s::text[]) "
            params.append(predicates)
        params.append(aid)
        params.append(max_hop)
        if predicates:
            params.append(predicates)
        params.extend([pool_size, edge_cap])

        sql = (
            """
            WITH RECURSIVE walk AS (
              SELECT
                s.canonical AS subject,
                kg.predicate,
                o.canonical AS obj,
                kg.object_id AS object_id,
                1 AS hop,
                ARRAY[kg.subject_id, kg.object_id]::uuid[] AS path
              FROM knowledge_graph kg
              JOIN entities s ON s.id = kg.subject_id
              JOIN entities o ON o.id = kg.object_id
              WHERE kg.agent_id = %s
                AND kg.subject_id = %s::uuid
            """
            + anchor_pred
            + """
              UNION ALL
              SELECT
                su.canonical AS subject,
                kg.predicate,
                ou.canonical AS obj,
                kg.object_id AS object_id,
                walk.hop + 1 AS hop,
                walk.path || kg.object_id AS path
              FROM walk
              JOIN knowledge_graph kg ON kg.subject_id = walk.object_id
              JOIN entities su ON su.id = kg.subject_id
              JOIN entities ou ON ou.id = kg.object_id
              WHERE kg.agent_id = %s
                AND walk.hop < %s
                AND NOT (kg.object_id = ANY(walk.path))
            """
            + recursive_pred
            + """
            ),
            pool AS (
              SELECT
                subject,
                predicate,
                obj,
                hop,
                (1.0 / hop::double precision) AS score
              FROM walk
              ORDER BY hop, subject, predicate
              LIMIT %s
            )
            SELECT subject, predicate, obj, hop
            FROM pool
            ORDER BY -ln(random()) / GREATEST(score, 0.01)
            LIMIT %s
            """
        )
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        return [
            {
                "subject": row[0],
                "predicate": row[1],
                "object": row[2],
                "hop": int(row[3]),
            }
            for row in rows
        ]

    def query_graph(
        self,
        subject: str | None = None,
        predicate: str | None = None,
        object_: str | None = None,
        limit: int = 50,
        agent: str | None = None,
    ) -> list[tuple]:
        """Structural search over entity text (ILIKE) and predicate."""
        aid = settings.default_agent_id if agent is None else agent
        lim = max(1, min(int(limit), 500))
        where_parts: list[str] = ["kg.agent_id = %s"]
        params: list = [aid]
        if subject and str(subject).strip():
            pat = f"%{subject.strip()}%"
            where_parts.append(
                "("
                "s.canonical ILIKE %s OR EXISTS ("
                "SELECT 1 FROM jsonb_array_elements_text(COALESCE(s.aliases, '[]'::jsonb)) sx(val) "
                "WHERE sx.val ILIKE %s)"
                ")"
            )
            params.extend([pat, pat])
        if predicate and str(predicate).strip():
            where_parts.append("kg.predicate ILIKE %s")
            params.append(f"%{predicate.strip()}%")
        if object_ and str(object_).strip():
            pat_o = f"%{object_.strip()}%"
            where_parts.append(
                "("
                "o.canonical ILIKE %s OR EXISTS ("
                "SELECT 1 FROM jsonb_array_elements_text(COALESCE(o.aliases, '[]'::jsonb)) ox(val) "
                "WHERE ox.val ILIKE %s)"
                ")"
            )
            params.extend([pat_o, pat_o])
        where_sql = (" AND ".join(where_parts)) if where_parts else "TRUE"
        params.append(lim)

        sql = f"""
            SELECT kg.subject, kg.predicate, kg.object
            FROM knowledge_graph kg
            JOIN entities s ON s.id = kg.subject_id
            JOIN entities o ON o.id = kg.object_id
            WHERE {where_sql}
            ORDER BY kg.created_at DESC NULLS LAST, kg.id
            LIMIT %s
            """

        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                return cur.fetchall()


_default_self_reflection: "SelfReflection | None" = None


def get_default_self_reflection() -> "SelfReflection":
    """Process-wide singleton SelfReflection (shared pool)."""
    global _default_self_reflection
    if _default_self_reflection is None:
        with _singleton_lock:
            if _default_self_reflection is None:
                _default_self_reflection = SelfReflection()
    return _default_self_reflection


PROTECTED_DIARY_EXACT = frozenset({"I am."})


def _is_protected_diary_content(content: str) -> bool:
    return (content or "").strip() in PROTECTED_DIARY_EXACT


class SelfReflection:
    """First-person agent narrative store — diary, not third-person KG."""

    def __init__(self, pool=None):
        self.pool = pool or get_pool()
        self._embed = AgentMemory(pool=self.pool)

    def get_vector(self, text: str, *, interactive: bool = False) -> list[float]:
        return self._embed.get_vector(text, interactive=interactive)

    def has_exact(self, content: str, agent: str | None = None) -> bool:
        """True when an identical content row exists (seed idempotency)."""
        text = (content or "").strip()
        if not text:
            return False
        aid = settings.default_agent_id if agent is None else agent
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT 1 FROM self_reflections
                    WHERE agent_id = %s AND content = %s
                    LIMIT 1
                    """,
                    (aid, text),
                )
                return cur.fetchone() is not None

    def save(
        self,
        text: str,
        *,
        importance: float = 0.5,
        agent: str | None = None,
        source: str = "reflect",
        source_ref: str | None = None,
        allow_missing_embedding: bool = False,
    ) -> bool:
        """Persist one reflection. Returns True on INSERT, False on skip/dedup/guard.

        ``allow_missing_embedding`` keeps the single write path usable when
        Ollama is down (heart outreach): the row is stored with a NULL vector and
        skipped by retrieval until a backfill fills it. Dedup cannot run without
        a vector, so that path is skipped too.
        """
        aid = settings.default_agent_id if agent is None else agent
        content = sanitize_memory_fact(text)
        if not content:
            _log.info("Self-reflection save skipped: injection pattern detected.")
            return False

        try:
            vector: str | None = _vector_param(self.get_vector(content))
        except Exception:
            if not allow_missing_embedding:
                raise
            vector = None
        imp = float(importance)

        signature: str | None = None
        try:
            from src.identity import get_signer

            signature = get_signer().sign(content)
        except Exception as exc:
            _log.warning("Self-reflection signature skipped: %s", exc)

        inserted_row = None
        with self.pool.connection() as conn:
            with conn.transaction():
                with conn.cursor() as cur:
                    # Same dedup-race guard as AgentMemory.save.
                    cur.execute(
                        "SELECT pg_advisory_xact_lock(hashtext(%s))",
                        (f"diary_save:{aid}",),
                    )
                    try:
                        if vector is None:
                            raise _SkipDiaryDedup
                        dedup_k = max(1, int(settings.memory_dedup_neighbors))
                        cosine_thr = float(settings.memory_dedup_cosine)
                        fuzzy_thr = float(settings.memory_dedup_fuzzy_cosine)
                        cur.execute(
                            """
                            SELECT
                              id,
                              content,
                              COALESCE(importance, %s) AS importance,
                              (1 - (embedding <=> %s::vector)) AS similarity
                            FROM self_reflections
                            WHERE agent_id = %s
                            ORDER BY embedding <=> %s::vector
                            LIMIT %s
                            """,
                            (
                                settings.memory_dedup_base_importance,
                                vector,
                                aid,
                                vector,
                                dedup_k,
                            ),
                        )
                        row = None
                        for candidate in cur.fetchall():
                            if not candidate or candidate[3] is None:
                                continue
                            sim = float(candidate[3])
                            existing_content = str(candidate[1] or "")
                            reason = _memory_dedup_reason(
                                content,
                                existing_content,
                                sim,
                                cosine_thr=cosine_thr,
                                fuzzy_thr=fuzzy_thr,
                            )
                            if reason:
                                row = candidate
                                break
                        if row is not None:
                            _log.info(
                                "Self-reflection dedup: similarity=%.3f, skip INSERT.",
                                float(row[3]),
                            )
                            return False
                    except _SkipDiaryDedup:
                        pass
                    except Exception as e:
                        _log.warning(
                            "Self-reflection dedup failed; aborting save: %s", e
                        )
                        raise

                    cur.execute(
                        """
                        INSERT INTO self_reflections (
                          agent_id, content, embedding, importance, signature,
                          source, source_ref
                        )
                        VALUES (%s, %s, %s::vector, %s, %s, %s, %s)
                        RETURNING id
                        """,
                        (aid, content, vector, imp, signature, source, source_ref),
                    )
                    inserted_row = cur.fetchone()

        if inserted_row:
            _log.info("Self-reflection saved: %s...", content[:50])
            return True
        return False

    def search(
        self,
        query: str,
        top_k: int = 3,
        agent: str | None = None,
        *,
        query_vector: list[float] | None = None,
        interactive: bool = False,
    ) -> list[tuple[str, float, datetime]]:
        """Semantic search over self-reflections; returns (content, score, created_at)."""
        aid = settings.default_agent_id if agent is None else agent
        limit = max(1, min(int(top_k), 50))
        try:
            if query_vector is not None:
                vector = _vector_param(query_vector)
            else:
                vector = _vector_param(self.get_vector(query, interactive=interactive))
        except Exception as e:
            _log.warning("Self-reflection search skipped (embed failed): %s", e)
            mark_embed_unavailable(str(e))
            return []

        sql, params = _self_reflection_search_sql(
            vector=vector,
            agent_id=aid,
            limit=limit,
            with_ids=False,
        )
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        return [(str(r[0]), float(r[1]), r[2]) for r in rows]

    def latest(
        self,
        n: int = 5,
        agent: str | None = None,
    ) -> list[tuple[str, datetime]]:
        """Most recent reflections for this agent, newest first."""
        aid = settings.default_agent_id if agent is None else agent
        limit = max(1, min(int(n), 50))
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT content, created_at
                    FROM self_reflections
                    WHERE agent_id = %s
                    ORDER BY created_at DESC NULLS LAST, id DESC
                    LIMIT %s
                    """,
                    (aid, limit),
                )
                rows = cur.fetchall()
        return [(str(r[0]), r[1]) for r in rows]

    def latest_with_ids(
        self,
        n: int = 5,
        agent: str | None = None,
    ) -> list[tuple[str, str, float, datetime]]:
        """Most recent reflections with ids (newest first); score=1.0 placeholder."""
        aid = settings.default_agent_id if agent is None else agent
        limit = max(1, min(int(n), 50))
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id::text, content, created_at
                    FROM self_reflections
                    WHERE agent_id = %s
                    ORDER BY created_at DESC NULLS LAST, id DESC
                    LIMIT %s
                    """,
                    (aid, limit),
                )
                rows = cur.fetchall()
        return [(str(r[0]), str(r[1]), 1.0, r[2]) for r in rows if r and r[1]]

    def recent_rows_with_embeddings(
        self,
        *,
        n: int = 20,
        agent: str | None = None,
    ) -> list[tuple[str, str, list[float]]]:
        """Newest diary rows as (id, content, embedding) for echo checks."""
        aid = settings.default_agent_id if agent is None else agent
        limit = max(1, min(int(n), 100))
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id::text, content, embedding
                    FROM self_reflections
                    WHERE agent_id = %s
                    ORDER BY created_at DESC NULLS LAST, id DESC
                    LIMIT %s
                    """,
                    (aid, limit),
                )
                rows = cur.fetchall()
        out: list[tuple[str, str, list[float]]] = []
        for row in rows:
            if not row or not row[0] or not row[1]:
                continue
            content = str(row[1])
            if content.strip() in PROTECTED_DIARY_EXACT:
                continue
            emb = _coerce_stored_embedding(row[2])
            if not emb:
                continue
            out.append((str(row[0]), content, emb))
        return out

    def search_with_ids(
        self,
        query: str,
        top_k: int = 3,
        agent: str | None = None,
        *,
        query_vector: list[float] | None = None,
        interactive: bool = False,
    ) -> list[tuple[str, str, float, datetime]]:
        """Semantic search; returns (id, content, score, created_at)."""
        aid = settings.default_agent_id if agent is None else agent
        limit = max(1, min(int(top_k), 50))
        try:
            if query_vector is not None:
                vector = _vector_param(query_vector)
            else:
                vector = _vector_param(self.get_vector(query, interactive=interactive))
        except Exception as e:
            _log.warning("Self-reflection search skipped (embed failed): %s", e)
            mark_embed_unavailable(str(e))
            return []

        sql, params = _self_reflection_search_sql(
            vector=vector,
            agent_id=aid,
            limit=limit,
            with_ids=True,
        )
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        return [(str(r[0]), str(r[1]), float(r[2]), r[3]) for r in rows if r and r[1]]

    def replace(
        self,
        reflection_id: str,
        new_content: str,
        *,
        agent: str | None = None,
    ) -> str:
        """Replace one diary row. Returns replaced | not_found | protected | injection_blocked."""
        aid = settings.default_agent_id if agent is None else agent
        rid = (reflection_id or "").strip()
        if not rid:
            return "not_found"
        try:
            rid = str(UUID(rid))
        except ValueError:
            return "not_found"
        content = sanitize_memory_fact(new_content)
        if not content:
            _log.info("Self-reflection replace skipped: injection pattern detected.")
            return "injection_blocked"

        signature: str | None = None
        try:
            from src.identity import get_signer

            signature = get_signer().sign(content)
        except Exception as exc:
            _log.warning("Self-reflection replace signature skipped: %s", exc)

        vector = _vector_param(self.get_vector(content))
        # One transaction: the protection check and the write see the same row.
        # A row that becomes a protected seed between check and write cannot slip
        # through, because it is locked FOR UPDATE until the transaction ends.
        with self.pool.connection() as conn:
            with conn.transaction():
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT content FROM self_reflections
                        WHERE id = %s::uuid AND agent_id = %s
                        FOR UPDATE
                        """,
                        (rid, aid),
                    )
                    row = cur.fetchone()
                    if not row:
                        return "not_found"
                    if _is_protected_diary_content(str(row[0] or "")):
                        _log.info(
                            "Self-reflection replace blocked: protected seed row."
                        )
                        return "protected"
                    cur.execute(
                        """
                        UPDATE self_reflections
                        SET content = %s, embedding = %s::vector, signature = %s
                        WHERE id = %s::uuid AND agent_id = %s
                        """,
                        (content, vector, signature, rid, aid),
                    )
                    updated = cur.rowcount > 0
        if updated:
            log_event(
                "self_reflection.replace",
                reflection_id=rid,
                len=len(content),
                preview=_memory_preview(content),
            )
            return "replaced"
        return "not_found"

    def delete(
        self,
        reflection_id: str,
        *,
        agent: str | None = None,
    ) -> str:
        """Delete one diary row. Returns deleted | not_found | protected."""
        aid = settings.default_agent_id if agent is None else agent
        rid = (reflection_id or "").strip()
        if not rid:
            return "not_found"
        try:
            rid = str(UUID(rid))
        except ValueError:
            return "not_found"

        # Same transaction discipline as replace(): lock the row, check, delete.
        with self.pool.connection() as conn:
            with conn.transaction():
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT content FROM self_reflections
                        WHERE id = %s::uuid AND agent_id = %s
                        FOR UPDATE
                        """,
                        (rid, aid),
                    )
                    row = cur.fetchone()
                    if not row:
                        return "not_found"
                    if _is_protected_diary_content(str(row[0] or "")):
                        return "protected"
                    cur.execute(
                        """
                        DELETE FROM self_reflections
                        WHERE id = %s::uuid AND agent_id = %s
                        """,
                        (rid, aid),
                    )
                    deleted = cur.rowcount > 0
        if deleted:
            log_event("self_reflection.delete", reflection_id=rid)
            return "deleted"
        return "not_found"


def _memory_cli_main() -> None:
    """CLI demo invoked from ``python -m src.memory`` (also unit-tested with mocks)."""
    mem = AgentMemory()
    mem.save("Alice bought an M5 Max 128GB and configured Postgres 18", "episodic")

    print("\nSearch check:")
    results = mem.search("What hardware does the owner use?")
    for mid, content, score in results:
        print(f"[{score:.2f}] id={mid} {content}")


if __name__ == "__main__":  # pragma: no cover
    _memory_cli_main()
