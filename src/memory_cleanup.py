"""Read-only memory cleanup planning — shared by ritual tool and operator scripts."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from src.config import settings
from src.memory import _is_text_near_duplicate


@dataclass(frozen=True)
class CleanupCandidate:
    memory_id: str
    preview: str
    reason: str
    action: str  # delete | merge | edit


class _UnionFind:
    def __init__(self) -> None:
        self._parent: dict[UUID, UUID] = {}

    def add(self, node: UUID) -> None:
        self._parent.setdefault(node, node)

    def find(self, node: UUID) -> UUID:
        parent = self._parent[node]
        if parent != node:
            self._parent[node] = self.find(parent)
        return self._parent[node]

    def union(self, a: UUID, b: UUID) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[rb] = ra


def _pair_is_duplicate(
    content_a: str,
    content_b: str,
    cosine: float,
    *,
    cosine_thr: float,
    fuzzy_thr: float,
) -> bool:
    if cosine >= cosine_thr:
        return True
    if cosine < fuzzy_thr:
        return False
    return _is_text_near_duplicate(content_a, content_b) or _is_text_near_duplicate(
        content_b, content_a
    )


def _fetch_rows(conn: Any, agent_id: str) -> list[dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, agent_id, content, COALESCE(importance, %s) AS importance, created_at
            FROM memories
            WHERE agent_id = %s
            ORDER BY created_at
            """,
            (settings.memory_dedup_base_importance, agent_id),
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]


def _fetch_candidate_pairs(
    conn: Any, agent_id: str, *, fuzzy_thr: float
) -> list[tuple]:
    """Near-duplicate pairs via per-row ANN neighbors (HNSW) instead of a full O(n²) self-join."""
    neighbors = max(10, int(settings.memory_dedup_neighbors))
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT
              a.id,
              b.id,
              a.content,
              b.content,
              b.cosine
            FROM memories a
            JOIN LATERAL (
              SELECT
                m.id,
                m.content,
                1 - (m.embedding <=> a.embedding) AS cosine
              FROM memories m
              WHERE m.agent_id = a.agent_id
                AND m.embedding IS NOT NULL
                AND m.id <> a.id
              ORDER BY m.embedding <=> a.embedding
              LIMIT %s
            ) b ON b.id > a.id
            WHERE a.agent_id = %s
              AND a.embedding IS NOT NULL
              AND b.cosine >= %s
            """,
            (neighbors, agent_id, fuzzy_thr),
        )
        return cur.fetchall()


def _pick_winner(members: list[dict[str, Any]]) -> dict[str, Any]:
    def _key(row: dict[str, Any]) -> tuple:
        imp = float(row.get("importance") or 0.0)
        created = row.get("created_at")
        if isinstance(created, datetime):
            ts = created.timestamp()
        else:
            ts = 0.0
        return (imp, ts)

    return max(members, key=_key)


def plan_duplicate_merges(
    conn: Any, *, agent_id: str
) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    """Return (winner, losers) for each near-duplicate cluster (read-only)."""
    cosine_thr = float(settings.memory_dedup_cosine)
    fuzzy_thr = float(settings.memory_dedup_fuzzy_cosine)
    rows = _fetch_rows(conn, agent_id)
    by_id = {row["id"]: row for row in rows}
    uf = _UnionFind()
    for row in rows:
        uf.add(row["id"])

    for id_a, id_b, content_a, content_b, cosine in _fetch_candidate_pairs(
        conn, agent_id, fuzzy_thr=fuzzy_thr
    ):
        if _pair_is_duplicate(
            str(content_a or ""),
            str(content_b or ""),
            float(cosine),
            cosine_thr=cosine_thr,
            fuzzy_thr=fuzzy_thr,
        ):
            uf.union(id_a, id_b)

    clusters: dict[UUID, list[UUID]] = defaultdict(list)
    for row_id in by_id:
        clusters[uf.find(row_id)].append(row_id)

    plans: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for member_ids in clusters.values():
        if len(member_ids) < 2:
            continue
        members = [by_id[mid] for mid in member_ids]
        winner = _pick_winner(members)
        losers = [m for m in members if m["id"] != winner["id"]]
        plans.append((winner, losers))
    return plans


def _stale_reason(row: dict[str, Any], *, ttl_days: int, min_importance: float) -> str:
    created = row.get("created_at")
    imp = float(row.get("importance") or settings.memory_dedup_base_importance)
    if isinstance(created, datetime):
        age_days = (datetime.now(created.tzinfo) - created).days
    else:
        age_days = 0
    if age_days >= ttl_days:
        return f"older than {ttl_days} days (MEMORY_TTL_DAYS)"
    return (
        f"low importance ({imp:.2f} < {min_importance}) "
        f"and older than 30 days (MEMORY_FORGET_MIN_IMPORTANCE)"
    )


def fetch_stale_candidates(
    conn: Any,
    *,
    agent_id: str,
    limit: int,
) -> list[dict[str, Any]]:
    """Rows eligible for TTL / low-importance cleanup (same rules as scripts/forget.py)."""
    ttl_days = int(settings.memory_ttl_days)
    min_importance = float(settings.memory_forget_min_importance)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, content, COALESCE(importance, %s) AS importance, created_at
            FROM memories
            WHERE agent_id = %s
              AND memory_type <> 'identity'
              AND content NOT LIKE %s
              AND (
                created_at < now() - (%s * INTERVAL '1 day')
                OR (
                  COALESCE(importance, %s) < %s
                  AND created_at < now() - INTERVAL '30 days'
                )
              )
            ORDER BY created_at ASC
            LIMIT %s
            """,
            (
                settings.memory_dedup_base_importance,
                agent_id,
                "%[AFFIRMATION signed:%",
                ttl_days,
                settings.memory_dedup_base_importance,
                min_importance,
                limit,
            ),
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]


def preview_content(text: str, *, max_len: int = 120) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 1] + "…"


def collect_cleanup_candidates(
    conn: Any,
    *,
    agent_id: str,
    limit: int = 20,
) -> list[CleanupCandidate]:
    """Merge duplicate losers first, then stale rows, up to ``limit`` total."""
    ttl_days = int(settings.memory_ttl_days)
    min_importance = float(settings.memory_forget_min_importance)
    out: list[CleanupCandidate] = []
    seen: set[str] = set()

    for winner, losers in plan_duplicate_merges(conn, agent_id=agent_id):
        winner_id = str(winner["id"])
        for loser in losers:
            lid = str(loser["id"])
            if lid in seen:
                continue
            out.append(
                CleanupCandidate(
                    memory_id=lid,
                    preview=preview_content(str(loser.get("content") or "")),
                    reason=f"near-duplicate of {winner_id}",
                    action="merge",
                )
            )
            seen.add(lid)
            if len(out) >= limit:
                return out

    remaining = limit - len(out)
    if remaining <= 0:
        return out

    for row in fetch_stale_candidates(conn, agent_id=agent_id, limit=remaining):
        lid = str(row["id"])
        if lid in seen:
            continue
        out.append(
            CleanupCandidate(
                memory_id=lid,
                preview=preview_content(str(row.get("content") or "")),
                reason=_stale_reason(
                    row, ttl_days=ttl_days, min_importance=min_importance
                ),
                action="delete",
            )
        )
        seen.add(lid)
        if len(out) >= limit:
            break
    return out


def ensure_memories_archive_table(cur: Any) -> None:
    """Lazy-create ``memories_archive`` (memories snapshot + ``archived_at``)."""
    cur.execute("""
        CREATE TABLE IF NOT EXISTS memories_archive (
            LIKE memories INCLUDING ALL
        )
        """)
    cur.execute("""
        ALTER TABLE memories_archive
        ADD COLUMN IF NOT EXISTS archived_at TIMESTAMPTZ
        """)


def parse_memory_ids(raw: str) -> list[str]:
    return [part.strip() for part in (raw or "").split(",") if part.strip()]


def parse_uuid_or_none(raw: str) -> str | None:
    """Return canonical UUID string or None when invalid."""
    text = (raw or "").strip()
    if not text:
        return None
    try:
        return str(UUID(text))
    except ValueError:
        return None


MemoryCurationStatus = Literal["not_found", "protected", "editable"]


def memory_curation_status(
    conn: Any,
    *,
    agent_id: str,
    memory_id: str,
) -> MemoryCurationStatus:
    """Whether an agent-scoped memory row may be edited or forgotten."""
    uid = parse_uuid_or_none(memory_id)
    if not uid:
        return "not_found"
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 1 FROM memories
            WHERE id = %s::uuid AND agent_id = %s
            """,
            (uid, agent_id),
        )
        if not cur.fetchone():
            return "not_found"
    allowed, blocked = filter_deletable_memory_ids(conn, [uid])
    if uid in blocked and uid not in allowed:
        return "protected"
    return "editable"


def filter_deletable_memory_ids(
    conn: Any, ids: list[str]
) -> tuple[list[str], list[str]]:
    """Drop identity / affirmation rows — never eligible for agent-led cleanup."""
    if not ids:
        return [], []
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id::text
            FROM memories
            WHERE id = ANY(%s::uuid[])
              AND memory_type <> 'identity'
              AND content NOT LIKE %s
            """,
            (ids, "%[AFFIRMATION signed:%"),
        )
        allowed = [str(row[0]) for row in cur.fetchall()]
    allowed_set = set(allowed)
    blocked = [i for i in ids if i not in allowed_set]
    return allowed, blocked


def forget_agent_memory(
    conn: Any,
    *,
    agent_id: str,
    memory_id: str,
    apply: bool = True,
) -> str:
    """Agent-scoped forget. Returns forgotten | not_found | protected."""
    uid = parse_uuid_or_none(memory_id)
    if not uid:
        return "not_found"
    status = memory_curation_status(conn, agent_id=agent_id, memory_id=uid)
    if status == "not_found":
        return "not_found"
    if status == "protected":
        return "protected"
    result = forget_memory_ids(conn, [uid], apply=apply)
    if int(result.get("deleted") or 0) > 0:
        return "forgotten"
    return "not_found"


def forget_memory_ids(conn: Any, ids: list[str], *, apply: bool) -> dict[str, Any]:
    """Archive (when applying) and delete rows by id. Dry-run lists matches only."""
    if not ids:
        return {
            "mode": "ids",
            "dry_run": not apply,
            "requested": 0,
            "found": 0,
            "archived": 0,
            "deleted": 0,
            "ids_found": [],
            "ids_missing": [],
        }

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id::text
            FROM memories
            WHERE id = ANY(%s::uuid[])
            """,
            (ids,),
        )
        found = [str(row[0]) for row in cur.fetchall()]
        found_set = set(found)
        missing = [i for i in ids if i not in found_set]

        archived = 0
        deleted = 0
        if apply and found:
            ensure_memories_archive_table(cur)
            cur.execute(
                """
                INSERT INTO memories_archive
                SELECT m.*, now() AS archived_at
                FROM memories m
                WHERE m.id = ANY(%s::uuid[])
                """,
                (found,),
            )
            archived = int(cur.rowcount or 0)
            cur.execute(
                "DELETE FROM memories WHERE id = ANY(%s::uuid[])",
                (found,),
            )
            deleted = int(cur.rowcount or 0)
            conn.commit()

    return {
        "mode": "ids",
        "dry_run": not apply,
        "requested": len(ids),
        "found": len(found),
        "archived": archived,
        "deleted": deleted,
        "ids_found": found,
        "ids_missing": missing,
    }


__all__ = [
    "CleanupCandidate",
    "collect_cleanup_candidates",
    "ensure_memories_archive_table",
    "fetch_stale_candidates",
    "filter_deletable_memory_ids",
    "forget_agent_memory",
    "forget_memory_ids",
    "memory_curation_status",
    "parse_memory_ids",
    "parse_uuid_or_none",
    "plan_duplicate_merges",
    "preview_content",
]
