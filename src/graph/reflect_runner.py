"""Run deferred reflect_node outside the user-facing graph path."""

from __future__ import annotations

import logging
import time
from typing import Any

from src.agent_runner import get_graph_app, graph_invoke_config
from src.graph.bg_dedup import spawn_deduped_daemon
from src.graph.session_summary_runner import thread_summary_lock
from src.obs.deferred_job_tracker import track_deferred_job
from src.obs.logs import log_event

_log = logging.getLogger(__name__)


def reflect_due(state: dict[str, Any]) -> bool:
    """True when reflect_node would run (ignoring ``reflect_defer``)."""
    from src.graph.workflow import reflect_eligible

    return reflect_eligible(state)  # type: ignore[arg-type]


def run_reflect_if_due(*, thread_id: str, lock_held: bool = False) -> float | None:
    """Extract durable facts when due; return elapsed ms or None if skipped."""
    tid = (thread_id or "").strip()
    if not tid:
        return None
    from src.agent_mode import is_asleep

    if is_asleep():
        log_event("reflect.skip", reason="asleep")
        return None

    def _run() -> float | None:
        app = get_graph_app()
        config = graph_invoke_config(thread_id=tid)
        try:
            snap = app.get_state(config)
        except Exception as exc:
            _log.warning("reflect state read failed thread_id=%s err=%s", tid, exc)
            return None
        if not snap or not snap.values:
            return None
        state: dict[str, Any] = dict(snap.values)
        if not reflect_due(state):
            return None

        from src.agent_runner import llm_compute_gate
        from src.graph.workflow import reflect_node

        t0 = time.perf_counter()
        try:
            # Marker: deferred-only consistency gate reads this and may do DB retrieval.
            state["_reflect_deferred"] = True
            with llm_compute_gate(label="reflect"):
                result = reflect_node(state)  # type: ignore[arg-type]
        except Exception as exc:
            _log.warning("reflect deferred failed thread_id=%s err=%s", tid, exc)
            return None
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if not result:
            log_event(
                "node.timing",
                node="reflect_node",
                elapsed_ms=round(elapsed_ms, 1),
                skipped=True,
                deferred=True,
            )
            return None

        for attempt in range(2):
            try:
                app.update_state(config, result)
                break
            except Exception as exc:
                if attempt == 0:
                    _log.warning(
                        "reflect state write retry thread_id=%s err=%s", tid, exc
                    )
                    continue
                log_event(
                    "reflect.checkpoint_update_failed",
                    thread_id=tid,
                    err=str(exc)[:200],
                )
                _log.warning("reflect state write failed thread_id=%s err=%s", tid, exc)
                return None

        log_event(
            "reflect.deferred",
            elapsed_ms=round(elapsed_ms, 1),
            thread_id=tid,
        )
        log_event(
            "node.timing",
            node="reflect_node",
            elapsed_ms=round(elapsed_ms, 1),
            deferred=True,
        )
        return elapsed_ms

    with track_deferred_job(tid, "reflect"):
        if lock_held:
            return _run()
        with thread_summary_lock(thread_id=tid):
            return _run()


def spawn_reflect_if_pending(*, thread_id: str, reflect_pending: bool) -> None:
    """Fire-and-forget reflect after reply (API/CLI/Telegram when defer=1)."""
    if not reflect_pending:
        return
    tid = (thread_id or "").strip()
    if not tid:
        return

    spawn_deduped_daemon(
        job="reflect",
        thread_id=tid,
        target=lambda: run_reflect_if_due(thread_id=tid),
        thread_name=f"deferred-reflect-{tid}",
    )


def schedule_reflect_background(*, thread_id: str) -> bool:
    """Start deferred reflect in a daemon thread when due."""
    from src.config import settings

    tid = (thread_id or "").strip()
    if not tid:
        return False

    if not settings.reflect_defer:
        return False

    app = get_graph_app()
    config = graph_invoke_config(thread_id=tid)
    try:
        snap = app.get_state(config)
        state = dict(snap.values) if snap and snap.values else {}
    except Exception as exc:
        _log.warning("reflect schedule skipped thread_id=%s err=%s", tid, exc)
        return False
    if not state or not reflect_due(state):
        return False

    spawn_reflect_if_pending(thread_id=tid, reflect_pending=True)
    return True


__all__ = [
    "reflect_due",
    "run_reflect_if_due",
    "schedule_reflect_background",
    "spawn_reflect_if_pending",
]
