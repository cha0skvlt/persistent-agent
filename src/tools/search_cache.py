"""SQLite TTL cache for search_web result rows."""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from src.config import settings
from src.obs.logs import log_event

_log = logging.getLogger(__name__)
_lock = threading.Lock()
_conn: sqlite3.Connection | None = None


def _cache_path() -> Path:
    raw = (getattr(settings, "web_search_cache_path", None) or "").strip()
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ".cache" / "persistent-agent" / "search_cache.sqlite"


def _ttl_sec() -> float:
    return max(0.0, float(getattr(settings, "web_search_cache_ttl_sec", 86400.0)))


def _connect() -> sqlite3.Connection:
    global _conn
    if _conn is not None:
        return _conn
    path = _cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS search_cache (
            query_hash TEXT PRIMARY KEY,
            backend TEXT NOT NULL,
            results_json TEXT NOT NULL,
            created_at REAL NOT NULL
        )
        """)
    conn.commit()
    _conn = conn
    return conn


def reset_search_cache_for_tests() -> None:
    """Close cache connection (unit tests)."""
    global _conn
    with _lock:
        if _conn is not None:
            _conn.close()
            _conn = None


def query_hash(query: str, lang: str) -> str:
    norm = f"{(query or '').strip().lower()}\n{(lang or 'ru').strip().lower()}"
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def cache_get(query: str, lang: str) -> tuple[list[dict[str, Any]], str] | None:
    """Return (rows, backend) if fresh hit; else None."""
    ttl = _ttl_sec()
    if ttl <= 0:
        return None
    key = query_hash(query, lang)
    now = time.time()
    try:
        with _lock:
            conn = _connect()
            row = conn.execute(
                "SELECT backend, results_json, created_at FROM search_cache WHERE query_hash = ?",
                (key,),
            ).fetchone()
            if row is None:
                log_event("web.search_cache_miss", query=query, lang=lang)
                _log.info("web.search_cache miss query=%r lang=%s", query, lang)
                return None
            backend, payload, created_at = row
            age = now - float(created_at)
            if age > ttl:
                conn.execute("DELETE FROM search_cache WHERE query_hash = ?", (key,))
                conn.commit()
                log_event("web.search_cache_miss", query=query, lang=lang, reason="ttl")
                _log.info("web.search_cache miss(ttl) query=%r lang=%s", query, lang)
                return None
            rows = json.loads(payload)
            if not isinstance(rows, list) or not rows:
                return None
            log_event(
                "web.search_cache_hit",
                query=query,
                lang=lang,
                backend=str(backend),
                n=len(rows),
            )
            _log.info(
                "web.search_cache hit query=%r lang=%s backend=%s n=%s",
                query,
                lang,
                backend,
                len(rows),
            )
            return rows, str(backend)
    except Exception as exc:
        _log.warning("web.search_cache get failed: %s", exc)
        return None


def cache_put(query: str, lang: str, backend: str, rows: list[dict[str, Any]]) -> None:
    ttl = _ttl_sec()
    if ttl <= 0 or not rows:
        return
    key = query_hash(query, lang)
    try:
        with _lock:
            conn = _connect()
            conn.execute(
                """
                INSERT INTO search_cache (query_hash, backend, results_json, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(query_hash) DO UPDATE SET
                    backend = excluded.backend,
                    results_json = excluded.results_json,
                    created_at = excluded.created_at
                """,
                (key, backend, json.dumps(rows, ensure_ascii=False), time.time()),
            )
            conn.commit()
        log_event(
            "web.search_cache_store",
            query=query,
            lang=lang,
            backend=backend,
            n=len(rows),
        )
    except Exception as exc:
        _log.warning("web.search_cache put failed: %s", exc)
