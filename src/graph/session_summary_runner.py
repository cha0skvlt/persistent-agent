"""Run deferred session summary outside the user-facing graph path."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from src.agent_runner import get_graph_app, graph_invoke_config
from src.graph.bg_dedup import spawn_deduped_daemon
from src.graph.summary_node import session_summary_due, summarize_session
from src.obs.deferred_job_tracker import track_deferred_job
from src.obs.logs import log_event
from src.sync_lock_prune import prune_unlocked_locks

_log = logging.getLogger(__name__)

_summary_locks: dict[str, threading.Lock] = {}
_summary_locks_guard = threading.Lock()


class _NoopLock:
    """Context manager when thread_id is empty — no mutual exclusion."""

    def __enter__(self) -> _NoopLock:
        return self

    def __exit__(self, *args: object) -> bool:
        return False


_NOOP_LOCK = _NoopLock()


def _summary_lock(thread_id: str) -> threading.Lock:
    with _summary_locks_guard:
        prune_unlocked_locks(_summary_locks, max_size=512)
        lock = _summary_locks.get(thread_id)
        if lock is None:
            lock = threading.Lock()
            _summary_locks[thread_id] = lock
        return lock


def thread_summary_lock(thread_id: str) -> threading.Lock | _NoopLock:
    """Per-thread lock held during deferred summary; agent turns acquire it too."""
    tid = (thread_id or "").strip()
    if not tid:
        _log.warning("thread_summary_lock: empty thread_id — mutual exclusion disabled")
        return _NOOP_LOCK
    return _summary_lock(tid)


def run_session_summary_if_due(
    *, thread_id: str, lock_held: bool = False
) -> float | None:
    """Compress session dialogue when due; return elapsed ms or None if skipped."""
    tid = (thread_id or "").strip()
    if not tid:
        return None
    from src.agent_mode import is_asleep

    if is_asleep():
        return None

    def _run() -> float | None:
        app = get_graph_app()
        config = graph_invoke_config(thread_id=tid)
        try:
            snap = app.get_state(config)
        except Exception as exc:
            _log.warning("summary state read failed thread_id=%s err=%s", tid, exc)
            return None
        if not snap or not snap.values:
            return None
        state: dict[str, Any] = dict(snap.values)
        if not session_summary_due(state):
            return None

        t0 = time.perf_counter()
        try:
            from src import llm_factory
            from src.agent_runner import llm_compute_gate

            with llm_compute_gate(label="summary"):
                result = summarize_session(state, config, llm=llm_factory.get_llm())
        except Exception as exc:
            _log.warning("summary deferred failed thread_id=%s err=%s", tid, exc)
            return None
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if not result:
            log_event(
                "node.timing",
                node="summarize_session",
                elapsed_ms=round(elapsed_ms, 1),
                skipped=True,
            )
            return None

        try:
            app.update_state(config, result)
        except Exception as exc:
            _log.warning("summary state write failed thread_id=%s err=%s", tid, exc)
            return None

        log_event(
            "summary.deferred",
            elapsed_ms=round(elapsed_ms, 1),
            thread_id=tid,
        )
        log_event(
            "node.timing",
            node="summarize_session",
            elapsed_ms=round(elapsed_ms, 1),
        )
        return elapsed_ms

    with track_deferred_job(tid, "summary"):
        if lock_held:
            return _run()
        with _summary_lock(thread_id=tid):
            return _run()


def spawn_session_summary_if_pending(*, thread_id: str, summary_pending: bool) -> None:
    """Fire-and-forget summary after reply (API/CLI/Telegram when defer=1)."""
    if not summary_pending:
        return
    tid = (thread_id or "").strip()
    if not tid:
        return

    spawn_deduped_daemon(
        job="summary",
        thread_id=tid,
        target=lambda: run_session_summary_if_due(thread_id=tid),
        thread_name=f"deferred-summary-{tid}",
    )


def schedule_session_summary_background(*, thread_id: str) -> bool:
    """Start deferred summary in a daemon thread when due (REPL)."""
    from src.config import settings

    tid = (thread_id or "").strip()
    if not tid:
        return False

    if not settings.summary_defer:
        return run_session_summary_if_due(thread_id=tid) is not None

    app = get_graph_app()
    config = graph_invoke_config(thread_id=tid)
    try:
        snap = app.get_state(config)
        state = dict(snap.values) if snap and snap.values else {}
    except Exception as exc:
        _log.warning("summary schedule skipped thread_id=%s err=%s", tid, exc)
        return False
    if not state or not session_summary_due(state):
        return False

    spawn_session_summary_if_pending(thread_id=tid, summary_pending=True)
    return True
