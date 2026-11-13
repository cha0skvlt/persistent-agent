#!/usr/bin/env python3
"""Export accumulated DPO gold pairs to JSONL for local preference tuning."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import settings  # noqa: E402
from src.dpo.gold_store import export_dpo_jsonl, list_gold_pairs  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Include pairs from the last N days (default: 7).",
    )
    parser.add_argument(
        "--agent-id",
        default="",
        help="Filter by agent_id (default: DEFAULT_AGENT_ID from settings).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output JSONL path (required).",
    )
    args = parser.parse_args()
    agent_id = (args.agent_id or settings.default_agent_id or "agent").strip()
    rows = list_gold_pairs(since_days=args.days, agent_id=agent_id)
    body = export_dpo_jsonl(since_days=args.days, agent_id=agent_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(body, encoding="utf-8")
    print(f"Wrote {len(rows)} pair(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
