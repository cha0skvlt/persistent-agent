from __future__ import annotations

import hashlib
import json
import logging
from typing import TYPE_CHECKING
from uuid import UUID

from src.config import settings

_log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.memory import AgentMemory

_MAX_AUTO_ALIAS_LEN = 48


def must_resolve_canonical_only(name: str) -> bool:
    """True when ``name`` is the self anchor or another agent callsign/persona."""
    from src.speaker import subject_keys

    n = (name or "").strip().lower()
    if not n:
        return False
    return n in subject_keys()


def must_not_attach_as_alias(name: str) -> bool:
    """Block embedding-merge from appending agent/persona surface forms as aliases."""
    return must_resolve_canonical_only(name)


def is_assistant_surface(name: str) -> bool:
    from src.speaker import is_generic_assistant_surface

    return is_generic_assistant_surface(name)


def is_agent_persona_surface(name: str) -> bool:
    """Generic local-agent persona surface, not a separate world entity."""
    n = (name or "").strip().lower()
    if not n:
        return False
    return n in {"agent persona", "self persona"}


def requires_distinct_entity(name: str) -> bool:
    """Force canonical INSERT — avoid embedding merge onto protected agent/system names."""
    if must_resolve_canonical_only(name) or is_agent_persona_surface(name):
        return False
    n = (name or "").strip().lower()
    if not n:
        return False
    if any(p in n for p in ("black light", "hottentot")):
        return True
    return False


def _append_entity_alias(active_conn, entity_id: UUID, alias: str, agent: str) -> None:
    piece = json.dumps([alias], ensure_ascii=False)
    with active_conn.cursor() as cur:
        cur.execute(
            """
            UPDATE entities
            SET aliases = aliases || %s::jsonb
            WHERE id = %s::uuid AND agent_id = %s
              AND NOT EXISTS (
                SELECT 1 FROM jsonb_array_elements_text(aliases) AS t(val)
                WHERE lower(val) = lower(%s)
              )
            """,
            (piece, str(entity_id), agent, alias),
        )


def _maybe_attach_entity_alias(
    active_conn, entity_id: UUID, alias: str, agent: str
) -> None:
    """Append ``alias`` after embedding merge when safe (length + identity guard)."""
    if len(alias) <= _MAX_AUTO_ALIAS_LEN and not must_not_attach_as_alias(alias):
        _append_entity_alias(active_conn, entity_id, alias, agent)


def resolve_kg_entity(
    name: str,
    kind: str | None = None,
    *,
    mem: "AgentMemory | None" = None,
    agent: str | None = None,
    conn=None,
    embed_cache: dict[str, list[float]] | None = None,
) -> UUID:
    """Resolve an entity for KG triple persistence (self anchor vs world)."""
    raw = (name or "").strip()
    if not raw:
        raise ValueError("entity name must be non-empty")

    from src.speaker import speaker_label

    anchor = speaker_label()
    if anchor and (is_assistant_surface(raw) or is_agent_persona_surface(raw)):
        return resolve_entity(
            anchor,
            kind=kind,
            mem=mem,
            agent=agent,
            conn=conn,
            embed_cache=embed_cache,
            canonical_only=True,
        )
    return resolve_entity(
        raw,
        kind=kind,
        mem=mem,
        agent=agent,
        conn=conn,
        embed_cache=embed_cache,
        canonical_only=requires_distinct_entity(raw),
    )


def _lookup_entity_by_name(
    name: str,
    *,
    agent: str | None = None,
    conn=None,
    canonical_only: bool = False,
) -> UUID | None:
    """Read-only entity lookup by canonical name or alias (no insert, no embed merge)."""
    from src.memory import get_default_memory

    aid = settings.default_agent_id if agent is None else agent
    raw = (name or "").strip()
    if not raw:
        return None
    pool = get_default_memory().pool

    def _lookup_on_conn(active_conn) -> UUID | None:
        with active_conn.cursor() as cur:
            cur.execute(
                """
                SELECT id FROM entities
                WHERE agent_id = %s
                  AND lower(trim(canonical)) = lower(%s)
                LIMIT 1
                """,
                (aid, raw),
            )
            row = cur.fetchone()
            if row:
                return UUID(str(row[0]))
            if canonical_only:
                return None
            cur.execute(
                """
                SELECT e.id
                FROM entities e,
                     jsonb_array_elements_text(e.aliases) AS elem(val)
                WHERE e.agent_id = %s
                  AND lower(val) = lower(%s)
                LIMIT 1
                """,
                (aid, raw),
            )
            row = cur.fetchone()
            if row:
                return UUID(str(row[0]))
            return None

    if conn is not None:
        return _lookup_on_conn(conn)
    with pool.connection() as active_conn:
        return _lookup_on_conn(active_conn)


def lookup_kg_entity(
    name: str,
    *,
    agent: str | None = None,
    conn=None,
) -> UUID | None:
    """Resolve an entity for read-only KG tools (no insert on miss)."""
    raw = (name or "").strip()
    if not raw:
        return None

    from src.speaker import speaker_label

    anchor = speaker_label()
    if anchor and (is_assistant_surface(raw) or is_agent_persona_surface(raw)):
        return _lookup_entity_by_name(
            anchor,
            agent=agent,
            conn=conn,
            canonical_only=True,
        )
    return _lookup_entity_by_name(
        raw,
        agent=agent,
        conn=conn,
        canonical_only=requires_distinct_entity(raw),
    )


def triple_hash_from_ids(subject_id: UUID, predicate: str, object_id: UUID) -> str:
    """Matches SQL: md5(subject_id::text || '|' || predicate || '|' || object_id::text)."""
    return hashlib.md5(f"{subject_id}|{predicate}|{object_id}".encode()).hexdigest()


def _pg_vector_literal(vec: list[float]) -> str:
    if not isinstance(vec, list) or not vec:
        raise ValueError(f"Invalid embedding vector: {type(vec)}")
    return "[" + ",".join(str(float(x)) for x in vec) + "]"


def _embed_for_entity(
    mem: "AgentMemory",
    text: str,
    *,
    embed_cache: dict[str, list[float]] | None,
) -> list[float]:
    key = (text or "").strip().lower()
    if not key:
        raise ValueError("entity name must be non-empty")
    if embed_cache is not None and key in embed_cache:
        return embed_cache[key]
    vec = mem.get_vector(text)
    if embed_cache is not None:
        embed_cache[key] = vec
    return vec


def resolve_entity(
    name: str,
    kind: str | None = None,
    *,
    mem: "AgentMemory | None" = None,
    agent: str | None = None,
    conn=None,
    embed_cache: dict[str, list[float]] | None = None,
    canonical_only: bool = False,
) -> UUID:
    """
    Resolve surface form to canonical entity UUID (insert new entity if needed).
    Order: canonical (ci), alias array (ci), cosine vs embedded rows (if any), else INSERT.

    When ``canonical_only`` is True (e.g. the agent entity), only exact
    canonical names match; aliases and embedding merge are skipped so the anchor
    is never attached to an unrelated entity that merely lists the name as alias.
    """
    from src.memory import AgentMemory

    mem = mem or AgentMemory()
    pool = mem.pool
    raw = (name or "").strip()
    if not raw:
        raise ValueError("entity name must be non-empty")

    if must_resolve_canonical_only(raw):
        canonical_only = True

    aid = settings.default_agent_id if agent is None else agent
    threshold = float(getattr(settings, "kg_entity_resolve_threshold", 0.90))

    def _resolve_on_conn(active_conn) -> UUID:
        with active_conn.cursor() as cur:
            cur.execute(
                """
                SELECT id FROM entities
                WHERE agent_id = %s
                  AND lower(trim(canonical)) = lower(%s)
                LIMIT 1
                """,
                (aid, raw),
            )
            row = cur.fetchone()
            if row:
                return UUID(str(row[0]))

            if not canonical_only:
                cur.execute(
                    """
                    SELECT e.id
                    FROM entities e,
                         jsonb_array_elements_text(e.aliases) AS elem(val)
                    WHERE e.agent_id = %s
                      AND lower(val) = lower(%s)
                    LIMIT 1
                    """,
                    (aid, raw),
                )
                row = cur.fetchone()
                if row:
                    return UUID(str(row[0]))

                cur.execute(
                    """
                    SELECT 1 FROM entities
                    WHERE agent_id = %s AND embedding IS NOT NULL
                    LIMIT 1
                    """,
                    (aid,),
                )
                if cur.fetchone():
                    vec_lit = _pg_vector_literal(
                        _embed_for_entity(mem, raw, embed_cache=embed_cache)
                    )
                    cur.execute(
                        """
                        SELECT id, (1 - (embedding <=> %s::vector)) AS sim
                        FROM entities
                        WHERE agent_id = %s
                          AND embedding IS NOT NULL
                        ORDER BY embedding <=> %s::vector
                        LIMIT 1
                        """,
                        (vec_lit, aid, vec_lit),
                    )
                    row = cur.fetchone()
                    if row and row[1] is not None and float(row[1]) >= threshold:
                        eid = UUID(str(row[0]))
                        _maybe_attach_entity_alias(active_conn, eid, raw, aid)
                        return eid

            vec_lit = _pg_vector_literal(
                _embed_for_entity(mem, raw, embed_cache=embed_cache)
            )
            cur.execute(
                """
                INSERT INTO entities (agent_id, canonical, aliases, kind, embedding)
                VALUES (%s, %s, '[]'::jsonb, %s, %s::vector)
                ON CONFLICT (agent_id, (lower(canonical))) DO NOTHING
                RETURNING id
                """,
                (aid, raw, kind, vec_lit),
            )
            row = cur.fetchone()
            if row:
                return UUID(str(row[0]))

            cur.execute(
                """
                SELECT id FROM entities
                WHERE agent_id = %s
                  AND lower(trim(canonical)) = lower(%s)
                LIMIT 1
                """,
                (aid, raw),
            )
            row = cur.fetchone()
            if row:
                return UUID(str(row[0]))
            raise RuntimeError(
                f"resolve_entity: insert conflict but no row for {raw!r} (agent={aid})"
            )

    if conn is not None:
        return _resolve_on_conn(conn)

    with pool.connection() as active_conn:
        with active_conn.transaction():
            return _resolve_on_conn(active_conn)
