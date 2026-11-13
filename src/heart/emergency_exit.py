"""Manual diagnostic override tick."""

from __future__ import annotations

import json
import os
import sys

import psycopg

from src.config import settings
from src.heart.body import collect_body_metrics
from src.heart.heartbeat import sign_tick_payload
from src.heart.vitals import write_tick


def _pg_conninfo() -> str:
    url = settings.database_url
    if url.startswith("postgresql+psycopg://"):
        url = "postgresql://" + url.removeprefix("postgresql+psycopg://")
    return url


def main() -> int:
    body = collect_body_metrics()
    body["emergency_manual_override"] = True
    body["emergency_exit_note"] = "operator_acknowledged"

    status = "ok"
    sig = sign_tick_payload(body, status)
    try:
        conn = psycopg.connect(_pg_conninfo(), autocommit=True)
    except Exception as exc:
        print(f"DB connection failed: {exc}", file=sys.stderr)
        return 1

    try:
        tid = write_tick(
            conn,
            pid=os.getpid(),
            body_metrics=body,
            phylactery_status=status,
            signature=sig,
        )
    finally:
        conn.close()

    if tid is None:
        print("Failed to write vitals tick.", file=sys.stderr)
        return 1
    print(json.dumps({"event": "emergency_exit.tick", "tick_id": tid}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
