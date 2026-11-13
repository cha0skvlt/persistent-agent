#!/usr/bin/env python3
"""Rebind KG entity IDs from surface text (self-anchor vs world). Does not touch ``memories``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import psycopg  # noqa: E402

from src.config import settings  # noqa: E402
from src.kg.entities import (  # noqa: E402
    resolve_kg_entity,
    triple_hash_from_ids,
)
from src.memory import AgentMemory  # noqa: E402

# Obvious test / noise edges only (never persona lore).
_DELETE_TRIPLES = (("a", "is", "b"),)

# Predicates where subject is the local agent (abliteration / voice), not USER.
_AGENT_AS_SUBJECT_PREDICATES = frozenset({"prefers"})
_AGENT_AS_SUBJECT_OBJECTS = frozenset({"provocative language"})


def _audit(conn) -> list[str]:
    lines: list[str] = []
    aid = settings.default_agent_id

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT kg.id::text,
                   e1.canonical, kg.subject,
                   kg.predicate,
                   e2.canonical, kg.object
            FROM knowledge_graph kg
            JOIN entities e1 ON e1.id = kg.subject_id
            JOIN entities e2 ON e2.id = kg.object_id
            WHERE kg.agent_id = %s
            """,
            (aid,),
        )
        for kid, s_canon, _s_text, pred, _o_canon, _o_text in cur.fetchall():
            sc = (s_canon or "").strip().lower()
            if sc == "assistant":
                lines.append(f"ASSISTANT_SUBJ id={kid} {pred} (should be self-anchor)")

        for s, p, o in _DELETE_TRIPLES:
            cur.execute(
                """
                SELECT 1 FROM knowledge_graph
                WHERE agent_id = %s AND lower(subject) = lower(%s)
                  AND predicate = %s AND lower(object) = lower(%s)
                """,
                (aid, s, p, o),
            )
            if cur.fetchone():
                lines.append(f"TEST_TRIPLE {s} --{p}--> {o}")

    return lines


def _apply(conn, mem: AgentMemory) -> tuple[int, int, int]:
    aid = settings.default_agent_id
    rebound = 0
    deleted = 0
    skipped = 0

    with conn.cursor() as cur:
        for s, p, o in _DELETE_TRIPLES:
            cur.execute(
                """
                DELETE FROM knowledge_graph
                WHERE agent_id = %s AND lower(subject) = lower(%s)
                  AND predicate = %s AND lower(object) = lower(%s)
                """,
                (aid, s, p, o),
            )
            deleted += cur.rowcount

        cur.execute(
            """
            SELECT id::text, subject, predicate, object
            FROM knowledge_graph
            WHERE agent_id = %s
            ORDER BY id
            """,
            (aid,),
        )
        rows = cur.fetchall()

    embed_cache: dict[str, list[float]] = {}
    with conn.transaction():
        with conn.cursor() as cur:
            agent_anchor = (settings.self_anchor_entity or "").strip()
            for kid, s_text, pred, o_text in rows:
                s_raw = (s_text or "").strip()
                o_raw = (o_text or "").strip()
                if not s_raw or not o_raw:
                    skipped += 1
                    continue
                if (
                    agent_anchor
                    and pred in _AGENT_AS_SUBJECT_PREDICATES
                    and o_raw.strip().lower() in _AGENT_AS_SUBJECT_OBJECTS
                ):
                    s_raw = agent_anchor
                try:
                    sid = resolve_kg_entity(
                        s_raw,
                        mem=mem,
                        agent=aid,
                        conn=conn,
                        embed_cache=embed_cache,
                    )
                    oid = resolve_kg_entity(
                        o_raw,
                        mem=mem,
                        agent=aid,
                        conn=conn,
                        embed_cache=embed_cache,
                    )
                except Exception:
                    skipped += 1
                    continue
                th = triple_hash_from_ids(sid, pred, o_raw)
                cur.execute(
                    """
                    UPDATE knowledge_graph
                    SET subject_id = %s::uuid,
                        object_id = %s::uuid,
                        subject = %s,
                        object = %s,
                        triple_hash = %s
                    WHERE id = %s::uuid AND agent_id = %s
                    """,
                    (str(sid), str(oid), s_raw, o_raw, th, kid, aid),
                )
                if cur.rowcount:
                    rebound += 1

    conn.commit()
    return rebound, deleted, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    mem = AgentMemory()
    with psycopg.connect(settings.database_url) as conn:
        issues = _audit(conn)
        if not issues:
            print("KG self-anchor wiring looks consistent.")
            return 0
        print(f"Found {len(issues)} issue(s):")
        for line in issues:
            print(f"  {line}")
        if not args.apply:
            print("\nRe-run with --apply to rebind (memories table untouched).")
            return 0
        rebound, deleted, skipped = _apply(conn, mem)
        issues_after = _audit(conn)
        print(
            f"Applied: rebind={rebound} deleted_test={deleted} skipped={skipped} "
            f"remaining_issues={len(issues_after)}"
        )
        for line in issues_after[:20]:
            print(f"  still: {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
