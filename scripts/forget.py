from __future__ import annotations

import argparse
import json
import sys
from uuid import UUID

from src.config import settings
from src.db.pool import get_pool
from src.memory_cleanup import forget_memory_ids, parse_memory_ids


def _forget_stale(*, dry_run: bool, ttl_days: int | None) -> dict:
    ttl = int(ttl_days if ttl_days is not None else settings.memory_ttl_days)
    min_importance = float(settings.memory_forget_min_importance)

    # Scoped to the configured agent; identity/affirmation rows are never TTL-deleted
    # (same protection as the by-id path in memory_cleanup).
    where_sql = """
        agent_id = %s
        AND memory_type <> 'identity'
        AND content NOT LIKE '%%[AFFIRMATION signed:%%'
        AND (
          (
            created_at < now() - (%s * INTERVAL '1 day')
          )
          OR
          (
            COALESCE(importance, 0.5) < %s
            AND created_at < now() - INTERVAL '30 days'
          )
        )
    """
    params = (settings.default_agent_id, ttl, min_importance)

    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT count(*) FROM memories WHERE {where_sql}", params)
            candidates = int(cur.fetchone()[0] or 0)

            deleted = 0
            if not dry_run and candidates > 0:
                cur.execute(f"DELETE FROM memories WHERE {where_sql}", params)
                deleted = int(cur.rowcount or 0)
                conn.commit()

    return {
        "mode": "stale",
        "deleted": deleted,
        "candidates": candidates,
        "dry_run": bool(dry_run),
        "ttl_days": ttl,
        "min_importance": min_importance,
    }


def _validate_uuid_list(ids: list[str]) -> tuple[list[str], list[str]]:
    valid: list[str] = []
    invalid: list[str] = []
    for raw in ids:
        try:
            valid.append(str(UUID(raw)))
        except ValueError:
            invalid.append(raw)
    return valid, invalid


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Forget stale memories (TTL/low-importance) or delete by id (with archive)."
    )
    parser.add_argument(
        "--ids",
        default=None,
        help="Comma-separated memory UUIDs to archive/delete.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="With --ids: archive then delete. Without --ids: unchanged (use absence of --dry-run).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Stale mode only: report candidates without deleting.",
    )
    parser.add_argument("--ttl-days", type=int, default=None)
    args = parser.parse_args()

    if args.ids is not None:
        raw_ids = parse_memory_ids(args.ids)
        valid, invalid = _validate_uuid_list(raw_ids)
        if invalid:
            print(
                json.dumps(
                    {"error": "invalid_uuid", "invalid_ids": invalid},
                    ensure_ascii=False,
                ),
                file=sys.stderr,
            )
            return 1

        pool = get_pool()
        with pool.connection() as conn:
            result = forget_memory_ids(conn, valid, apply=bool(args.apply))
        if invalid:
            result["ids_invalid"] = invalid
        print(json.dumps(result, ensure_ascii=False))
        return 0

    result = _forget_stale(dry_run=bool(args.dry_run), ttl_days=args.ttl_days)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
