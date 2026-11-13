from __future__ import annotations

import argparse
import json
from datetime import datetime
from time import perf_counter
from typing import Any

import psycopg

from src.config import settings
from src.ollama_client import ollama_http_client


def _iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    try:
        return dt.isoformat()
    except Exception:
        return str(dt)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Healthcheck (Postgres + Ollama embeddings) with metrics."
    )
    parser.add_argument("--json", action="store_true", default=True)
    parser.add_argument("--no-json", action="store_false", dest="json")
    args = parser.parse_args()

    db_url = settings.database_url
    embed_model = settings.embed_model

    db_ok = False
    db_latency_ms: float | None = None
    counts: dict[str, int] = {
        "memories": 0,
        "knowledge_graph": 0,
        "conversations": 0,
        "entities": 0,
    }
    latest_memory_at: str | None = None
    db_err: str | None = None

    t0 = perf_counter()
    try:
        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
                db_ok = True
            db_latency_ms = (perf_counter() - t0) * 1000.0

            with conn.cursor() as cur:
                cur.execute("SELECT count(*) FROM memories")
                counts["memories"] = int(cur.fetchone()[0] or 0)
                cur.execute("SELECT count(*) FROM knowledge_graph")
                counts["knowledge_graph"] = int(cur.fetchone()[0] or 0)
                cur.execute("SELECT count(*) FROM conversations")
                counts["conversations"] = int(cur.fetchone()[0] or 0)
                try:
                    cur.execute("SELECT count(*) FROM entities")
                    counts["entities"] = int(cur.fetchone()[0] or 0)
                except Exception:
                    counts["entities"] = 0
                cur.execute("SELECT max(created_at) FROM memories")
                latest_memory_at = _iso(cur.fetchone()[0])
    except Exception as e:
        db_err = str(e)
        db_latency_ms = (perf_counter() - t0) * 1000.0

    ollama_ok = False
    embed_latency_ms: float | None = None
    ollama_err: str | None = None
    t1 = perf_counter()
    try:
        _ = ollama_http_client().embeddings(
            model=embed_model,
            prompt="ping",
            keep_alive=settings.llm_keep_alive,
        )
        ollama_ok = True
        embed_latency_ms = (perf_counter() - t1) * 1000.0
    except Exception as e:
        ollama_err = str(e)
        embed_latency_ms = (perf_counter() - t1) * 1000.0

    if not db_ok:
        status = "down"
        exit_code = 2
    elif not ollama_ok:
        status = "degraded"
        exit_code = 1
    else:
        status = "ok"
        exit_code = 0

    payload: dict[str, Any] = {
        "status": status,
        "db": {
            "ok": db_ok,
            "latency_ms": db_latency_ms,
            "counts": counts,
            "latest_memory_at": latest_memory_at,
            "error": db_err,
        },
        "ollama": {
            "ok": ollama_ok,
            "latency_ms": embed_latency_ms,
            "error": ollama_err,
            "model": embed_model,
        },
    }

    if args.json:
        print(json.dumps(payload, ensure_ascii=False))
    else:
        print(f"status={status}")
        db_lat_s = (
            f"{db_latency_ms:.1f}"
            if isinstance(db_latency_ms, (int, float))
            else "null"
        )
        emb_lat_s = (
            f"{embed_latency_ms:.1f}"
            if isinstance(embed_latency_ms, (int, float))
            else "null"
        )
        print(
            f"db.ok={db_ok} db.latency_ms={db_lat_s} db.counts={counts} latest_memory_at={latest_memory_at}"
        )
        print(
            f"ollama.ok={ollama_ok} ollama.latency_ms={emb_lat_s} model={embed_model}"
        )
        if db_err:
            print(f"db.error={db_err}")
        if ollama_err:
            print(f"ollama.error={ollama_err}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
