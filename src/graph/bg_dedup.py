"""Dedup fire-and-forget background graph jobs per (job, thread_id)."""

from __future__ import annotations

import threading
from collections.abc import Callable

_running: set[str] = set()
_guard = threading.Lock()


def _job_key(job: str, thread_id: str) -> str:
    return f"{(job or '').strip()}:{(thread_id or '').strip()}"


def try_start_bg_job(job: str, thread_id: str) -> bool:
    """Register a background job; return False if the same job already runs."""
    tid = (thread_id or "").strip()
    j = (job or "").strip()
    if not tid or not j:
        return False
    key = _job_key(j, tid)
    with _guard:
        if key in _running:
            return False
        _running.add(key)
        return True


def finish_bg_job(job: str, thread_id: str) -> None:
    tid = (thread_id or "").strip()
    j = (job or "").strip()
    if not tid or not j:
        return
    with _guard:
        _running.discard(_job_key(j, tid))


def spawn_deduped_daemon(
    *,
    job: str,
    thread_id: str,
    target: Callable[[], object],
    thread_name: str,
) -> bool:
    """Start a daemon thread unless the same job is already running for thread_id."""
    if not try_start_bg_job(job, thread_id):
        return False

    def _run() -> None:
        try:
            target()
        finally:
            finish_bg_job(job, thread_id)

    try:
        thread = threading.Thread(target=_run, name=thread_name, daemon=True)
        thread.start()
    except Exception:
        # A failed start must not leave the key registered forever: the next
        # attempt for this (job, thread_id) would be blocked with no runner.
        finish_bg_job(job, thread_id)
        raise
    return True


def reset_bg_dedup() -> None:
    """Clear in-flight registry (tests)."""
    with _guard:
        _running.clear()


__all__ = [
    "finish_bg_job",
    "reset_bg_dedup",
    "spawn_deduped_daemon",
    "try_start_bg_job",
]
