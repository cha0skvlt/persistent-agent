#!/usr/bin/env python3
"""Prune old LangGraph checkpoints and aged DPO gold pairs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.db.pool import get_pool  # noqa: E402


def _count_checkpoints(cur) -> int:
    cur.execute("SELECT count(*) FROM checkpoints")
    return int(cur.fetchone()[0] or 0)


def _prune_checkpoints(cur, *, keep_per_thread: int) -> int:
    keep = max(1, int(keep_per_thread))
    cur.execute(
        """
        WITH ranked AS (
          SELECT
            thread_id,
            checkpoint_ns,
            checkpoint_id,
            ROW_NUMBER() OVER (
              PARTITION BY thread_id, checkpoint_ns
              ORDER BY checkpoint_id DESC
            ) AS rn
          FROM checkpoints
        ),
        doomed AS (
          SELECT thread_id, checkpoint_ns, checkpoint_id
          FROM ranked
          WHERE rn > %s
        )
        DELETE FROM checkpoint_writes cw
        USING doomed d
        WHERE cw.thread_id = d.thread_id
          AND cw.checkpoint_ns = d.checkpoint_ns
          AND cw.checkpoint_id = d.checkpoint_id
        """,
        (keep,),
    )
    writes_deleted = int(cur.rowcount or 0)
    cur.execute(
        """
        WITH ranked AS (
          SELECT
            thread_id,
            checkpoint_ns,
            checkpoint_id,
            ROW_NUMBER() OVER (
              PARTITION BY thread_id, checkpoint_ns
              ORDER BY checkpoint_id DESC
            ) AS rn
          FROM checkpoints
        ),
        doomed AS (
          SELECT thread_id, checkpoint_ns, checkpoint_id
          FROM ranked
          WHERE rn > %s
        )
        DELETE FROM checkpoints c
        USING doomed d
        WHERE c.thread_id = d.thread_id
          AND c.checkpoint_ns = d.checkpoint_ns
          AND c.checkpoint_id = d.checkpoint_id
        """,
        (keep,),
    )
    checkpoints_deleted = int(cur.rowcount or 0)
    return checkpoints_deleted + writes_deleted


def _prune_gold_pairs(cur, *, gold_days: int) -> int:
    days = max(1, int(gold_days))
    cur.execute(
        """
        DELETE FROM dpo_gold_pairs
        WHERE created_at < now() - (%s * INTERVAL '1 day')
        """,
        (days,),
    )
    return int(cur.rowcount or 0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keep-per-thread",
        type=int,
        default=50,
        help="Keep the newest N checkpoints per (thread_id, checkpoint_ns).",
    )
    parser.add_argument(
        "--gold-days",
        type=int,
        default=90,
        help="Delete dpo_gold_pairs older than N days (0 = skip).",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT EXISTS (
                  SELECT 1
                  FROM information_schema.tables
                  WHERE table_schema = 'public'
                    AND table_name = 'checkpoints'
                )
                """)
            has_checkpoints = bool(cur.fetchone()[0])
            cur.execute("""
                SELECT EXISTS (
                  SELECT 1
                  FROM information_schema.tables
                  WHERE table_schema = 'public'
                    AND table_name = 'dpo_gold_pairs'
                )
                """)
            has_gold = bool(cur.fetchone()[0])

            before_ckpt = _count_checkpoints(cur) if has_checkpoints else 0
            ckpt_candidates = 0
            gold_candidates = 0
            if has_checkpoints:
                keep = max(1, int(args.keep_per_thread))
                cur.execute(
                    """
                    WITH ranked AS (
                      SELECT
                        ROW_NUMBER() OVER (
                          PARTITION BY thread_id, checkpoint_ns
                          ORDER BY checkpoint_id DESC
                        ) AS rn
                      FROM checkpoints
                    )
                    SELECT count(*) FROM ranked WHERE rn > %s
                    """,
                    (keep,),
                )
                ckpt_candidates = int(cur.fetchone()[0] or 0)

            if has_gold and int(args.gold_days) > 0:
                cur.execute(
                    """
                    SELECT count(*) FROM dpo_gold_pairs
                    WHERE created_at < now() - (%s * INTERVAL '1 day')
                    """,
                    (max(1, int(args.gold_days)),),
                )
                gold_candidates = int(cur.fetchone()[0] or 0)

            ckpt_deleted = 0
            gold_deleted = 0
            if not args.dry_run:
                if has_checkpoints and ckpt_candidates:
                    ckpt_deleted = _prune_checkpoints(
                        cur, keep_per_thread=args.keep_per_thread
                    )
                if has_gold and gold_candidates and int(args.gold_days) > 0:
                    gold_deleted = _prune_gold_pairs(cur, gold_days=args.gold_days)
                conn.commit()

            after_ckpt = _count_checkpoints(cur) if has_checkpoints else 0

    report = {
        "dry_run": bool(args.dry_run),
        "checkpoints_before": before_ckpt,
        "checkpoints_after": after_ckpt,
        "checkpoint_rows_pruned": ckpt_candidates if args.dry_run else ckpt_deleted,
        "gold_pairs_pruned": gold_candidates if args.dry_run else gold_deleted,
        "keep_per_thread": int(args.keep_per_thread),
        "gold_days": int(args.gold_days),
    }
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
