#!/usr/bin/env python3
"""Merge near-duplicate rows in ``memories``."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import psycopg  # noqa: E402

from src.config import settings  # noqa: E402
from src.memory import _is_text_near_duplicate  # noqa: E402
from src.obs.log_format import clip_words  # noqa: E402


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


def _fetch_rows(conn: psycopg.Connection, agent_id: str | None) -> list[dict[str, Any]]:
    params: list[Any] = []
    agent_clause = ""
    if agent_id:
        agent_clause = "WHERE agent_id = %s"
        params.append(agent_id)
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT id, agent_id, content, COALESCE(importance, %s), created_at
            FROM memories
            {agent_clause}
            ORDER BY created_at
            """,
            [settings.memory_dedup_base_importance, *params],
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]


def _fetch_candidate_pairs(
    conn: psycopg.Connection, agent_id: str | None, *, fuzzy_thr: float
) -> list[tuple]:
    params: list[Any] = [fuzzy_thr]
    agent_clause = ""
    if agent_id:
        agent_clause = "AND a.agent_id = %s AND b.agent_id = %s"
        params.extend([agent_id, agent_id])
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT
              a.id,
              b.id,
              a.content,
              b.content,
              1 - (a.embedding <=> b.embedding) AS cosine
            FROM memories a
            JOIN memories b ON a.id < b.id AND a.agent_id = b.agent_id
            WHERE a.embedding IS NOT NULL
              AND b.embedding IS NOT NULL
              AND 1 - (a.embedding <=> b.embedding) >= %s
            {agent_clause}
            """,
            params,
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


def plan_merge(
    conn: psycopg.Connection, *, agent_id: str | None = None
) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    """Return (winner, losers) for each duplicate cluster."""
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


def apply_merge(
    conn: psycopg.Connection, plans: list[tuple[dict[str, Any], list[dict[str, Any]]]]
) -> int:
    deleted = 0
    with conn.cursor() as cur:
        for winner, losers in plans:
            bonus = sum(float(loser.get("importance") or 0.0) for loser in losers)
            cur.execute(
                """
                UPDATE memories
                SET importance = LEAST(COALESCE(importance, %s) + %s, %s)
                WHERE id = %s
                """,
                (
                    settings.memory_dedup_base_importance,
                    bonus,
                    settings.memory_importance_cap,
                    winner["id"],
                ),
            )
            for loser in losers:
                cur.execute("DELETE FROM memories WHERE id = %s", (loser["id"],))
                deleted += 1
    conn.commit()
    return deleted


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge near-duplicate memories.")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Execute merges (default: dry-run only).",
    )
    parser.add_argument(
        "--agent",
        default=None,
        help=f"Limit to agent_id (default: all; app default is {settings.default_agent_id!r}).",
    )
    args = parser.parse_args()
    agent_id = args.agent or None

    print("=== Memory dedupe ===")
    print(
        f"thresholds: cosine≥{settings.memory_dedup_cosine} "
        f"fuzzy≥{settings.memory_dedup_fuzzy_cosine} "
        f"text_overlap≥{settings.memory_dedup_text_overlap}"
    )
    if agent_id:
        print(f"agent_id: {agent_id}")

    with psycopg.connect(settings.database_url) as conn:
        plans = plan_merge(conn, agent_id=agent_id)
        if not plans:
            print("No duplicate clusters found.")
            return 0

        total_delete = sum(len(losers) for _, losers in plans)
        print(f"Clusters: {len(plans)}  rows to delete: {total_delete}")
        for winner, losers in plans:
            preview = clip_words(str(winner.get("content") or ""))
            print(
                f"  keep {winner['id']}  "
                f"importance→bump  "
                f"delete {len(losers)}  "
                f'"{preview}"'
            )
            for loser in losers:
                print(
                    f"    - {loser['id']}  "
                    f"{clip_words(str(loser.get('content') or ''))}"
                )

        if not args.apply:
            print("\nDry-run only. Re-run with --apply to execute.")
            return 0

        deleted = apply_merge(conn, plans)
        print(f"\nApplied: deleted {deleted} duplicate row(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
