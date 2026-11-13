"""In-flight deferred graph jobs per thread_id (instrumentation only; no locking)."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Iterator

from src.obs.logs import log_event

_active: dict[str, set[str]] = {}
_guard = threading.Lock()


def deferred_job_active(thread_id: str) -> bool:
    """True when any deferred job is running for ``thread_id``."""
    tid = (thread_id or "").strip()
    if not tid:
        return False
    with _guard:
        return bool(_active.get(tid))


def active_deferred_jobs(thread_id: str) -> frozenset[str]:
    tid = (thread_id or "").strip()
    if not tid:
        return frozenset()
    with _guard:
        return frozenset(_active.get(tid, ()))


def live_turn_overlaps_deferred(thread_id: str) -> bool:
    """Sample before a live turn acquires ``thread_summary_lock``."""
    return deferred_job_active(thread_id)


@contextmanager
def track_deferred_job(thread_id: str, job: str) -> Iterator[None]:
    """Mark deferred reflect/summary/role_review active for overlap measurement."""
    tid = (thread_id or "").strip()
    j = (job or "").strip()
    if tid and j:
        with _guard:
            _active.setdefault(tid, set()).add(j)
        log_event("deferred.job.start", job=j, thread_id=tid)
    try:
        yield
    finally:
        if tid and j:
            last_for_thread = False
            with _guard:
                jobs = _active.get(tid)
                if jobs:
                    jobs.discard(j)
                    if not jobs:
                        del _active[tid]
                        last_for_thread = True
            log_event("deferred.job.end", job=j, thread_id=tid)
            if last_for_thread:
                _unload_models_if_asleep()


def _unload_models_if_asleep() -> None:
    """A job in flight when /make sleep hit reloads the model — drop it again."""
    try:
        from src.agent_mode import is_asleep, unload_ollama_models

        if is_asleep():
            unload_ollama_models()
    except Exception as exc:
        log_event("deferred.asleep_unload_failed", err=type(exc).__name__)


def reset_deferred_job_tracker() -> None:
    """Clear registry (tests)."""
    with _guard:
        _active.clear()


__all__ = [
    "active_deferred_jobs",
    "deferred_job_active",
    "live_turn_overlaps_deferred",
    "reset_deferred_job_tracker",
    "track_deferred_job",
]
