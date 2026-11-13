from __future__ import annotations

import logging
import threading

from psycopg_pool import ConnectionPool

from src.config import settings

_log = logging.getLogger(__name__)

_pool: ConnectionPool | None = None
_checkpoint_pool: ConnectionPool | None = None
_pool_lock = threading.Lock()


def _open_pool(*, min_size: int, max_size: int, label: str) -> ConnectionPool:
    pool = ConnectionPool(
        conninfo=settings.database_url,
        open=True,
        min_size=min_size,
        max_size=max_size,
        timeout=float(settings.db_pool_timeout),
        kwargs={"autocommit": True},
    )
    _log.debug(
        "DB %s pool initialized (min=%s max=%s).",
        label,
        min_size,
        max_size,
    )
    return pool


def get_pool() -> ConnectionPool:
    """Lazy singleton for memory, KG, conversations (app data)."""
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = _open_pool(
                    min_size=int(settings.db_pool_min),
                    max_size=int(settings.db_pool_max),
                    label="app",
                )
    return _pool


def get_checkpoint_pool() -> ConnectionPool:
    """Lazy singleton for LangGraph ``PostgresSaver`` only (isolated from app pool)."""
    global _checkpoint_pool
    if _checkpoint_pool is None:
        with _pool_lock:
            if _checkpoint_pool is None:
                _checkpoint_pool = _open_pool(
                    min_size=int(settings.db_checkpoint_pool_min),
                    max_size=int(settings.db_checkpoint_pool_max),
                    label="checkpoint",
                )
    return _checkpoint_pool


def close_pool() -> None:
    """Close app + checkpoint pools (tests/shutdown only)."""
    global _pool, _checkpoint_pool
    with _pool_lock:
        if _checkpoint_pool is not None:
            _checkpoint_pool.close()
            _checkpoint_pool = None
        if _pool is not None:
            _pool.close()
            _pool = None
