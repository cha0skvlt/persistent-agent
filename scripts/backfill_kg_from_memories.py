#!/usr/bin/env python3
"""Extract KG triples from recent memories via chat model."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import psycopg  # noqa: E402
from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402

from src.config import settings  # noqa: E402
from src.graph.workflow import (  # noqa: E402
    _is_self_referential_subject,
    _parse_reflect_json,
    _parse_reflect_triple,
)
from src.llm_factory import get_llm  # noqa: E402
from src.memory import get_default_memory  # noqa: E402
from src.obs.log_format import clip_words  # noqa: E402

_EXTRACT_SYS = "You return only valid JSON."
_EXTRACT_USER = """Extract one (subject, predicate, object) triple from the fact if there is an explicit relation.
If there is no triple, return {{"triple": null}}.
Otherwise: {{"triple": {{"s":"...","p":"...","o":"..."}}}}

Fact:
{fact}
"""


def _extract_triple(fact: str) -> tuple[str, str, str] | None:
    llm = get_llm()
    res = llm.invoke(
        [
            SystemMessage(content=_EXTRACT_SYS),
            HumanMessage(content=_EXTRACT_USER.format(fact=fact)),
        ]
    )
    raw = str(getattr(res, "content", "") or "")
    data = _parse_reflect_json(raw)
    triple = data.get("triple")
    if triple is None:
        return None
    return _parse_reflect_triple(triple)


def _fetch_memories(
    conn: psycopg.Connection, *, days: int, agent_id: str | None
) -> list[tuple[str, str]]:
    since = datetime.now(timezone.utc) - timedelta(days=days)
    params: list[object] = [since]
    agent_clause = ""
    if agent_id:
        agent_clause = "AND agent_id = %s"
        params.append(agent_id)
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT id::text, content
            FROM memories
            WHERE created_at >= %s
            {agent_clause}
            ORDER BY created_at
            """,
            params,
        )
        return [(str(r[0]), str(r[1] or "")) for r in cur.fetchall()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backfill knowledge_graph from recent memories."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write triples (default: dry-run).",
    )
    parser.add_argument("--days", type=int, default=30, help="Lookback window.")
    parser.add_argument(
        "--agent",
        default=None,
        help=f"agent_id filter (default: all; app default {settings.default_agent_id!r}).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max memories to process (0 = all in window).",
    )
    args = parser.parse_args()

    print("=== KG backfill from memories ===")
    print(f"model: {settings.reasoning_model}")
    print(f"window: {args.days} days  apply={args.apply}")

    mem = get_default_memory()
    candidates: list[tuple[str, str, str, str]] = []
    skipped_self = 0
    no_triple = 0

    with psycopg.connect(settings.database_url) as conn:
        rows = _fetch_memories(conn, days=args.days, agent_id=args.agent)
        if args.limit > 0:
            rows = rows[: args.limit]

        for mid, content in rows:
            if not content.strip():
                continue
            try:
                parsed = _extract_triple(content)
            except Exception as e:
                print(f"  skip {mid[:8]}…  LLM error: {e}")
                continue
            if parsed is None:
                no_triple += 1
                continue
            s, p, o = parsed
            if _is_self_referential_subject(s):
                skipped_self += 1
                print(f"  skip self  {s} {p} {o}")
                continue
            candidates.append((mid, s, p, o))

    print(f"memories scanned: {len(rows)}")
    print(f"no triple: {no_triple}  self-filtered: {skipped_self}")
    print(f"candidates: {len(candidates)}")
    for mid, s, p, o in candidates[:50]:
        print(f"  {mid[:8]}…  {clip_words(f'{s} | {p} | {o}')}")
    if len(candidates) > 50:
        print(f"  … +{len(candidates) - 50} more")

    if not args.apply:
        print("\nDry-run only. Re-run with --apply to write KG.")
        return 0

    inserted = 0
    deduped = 0
    for _mid, s, p, o in candidates:
        if mem.save_fact_to_graph(s, p, o, agent=args.agent):
            inserted += 1
        else:
            deduped += 1
    print(f"\nApplied: inserted={inserted} dedup={deduped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
