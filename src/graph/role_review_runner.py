"""Deferred role review for DPO gold pairs (after user-visible reply)."""

from __future__ import annotations

import logging
import time
from typing import Any

from src.agent_reply import extract_agent_reply_text
from src.config import settings
from src.dpo.gold_store import insert_gold_pair
from src.dpo.role_review import review_turn
from src.graph.bg_dedup import spawn_deduped_daemon
from src.graph.session_summary_runner import thread_summary_lock
from src.obs.deferred_job_tracker import track_deferred_job
from src.obs.logs import log_event
from src.utils import last_real_human_text

_log = logging.getLogger(__name__)


def role_review_due(state: dict[str, Any]) -> bool:
    """True when background role review should run for this turn."""
    if not settings.role_review:
        return False
    tc = int(state.get("turn_count", 0) or 0)
    if tc <= 0:
        return False
    every = max(1, int(settings.role_review_every_n))
    return tc % every == 0


def _last_turn_pair(state: dict[str, Any]) -> tuple[str, str]:
    """Return ``(user_text, agent_reply)`` for the latest completed turn."""
    msgs = list(state.get("messages") or [])
    user = last_real_human_text(msgs)
    reply = extract_agent_reply_text(msgs, preserve_thinking=False)
    if not user or not reply:
        return "", ""
    return user.strip(), reply.strip()


def run_role_review_for_thread(*, thread_id: str) -> float | None:
    """Critique last turn; persist gold pair when role_failed. Return elapsed ms."""
    if not settings.role_review:
        return None
    tid = (thread_id or "").strip()
    if not tid:
        return None
    from src.agent_mode import is_asleep

    if is_asleep():
        return None

    from src.agent_runner import get_graph_app, graph_invoke_config

    with track_deferred_job(tid, "role_review"):
        with thread_summary_lock(thread_id=tid):
            app = get_graph_app()
            config = graph_invoke_config(thread_id=tid)
            try:
                snap = app.get_state(config)
            except Exception as exc:
                _log.warning(
                    "role_review state read failed thread_id=%s err=%s", tid, exc
                )
                return None
            if not snap or not snap.values:
                return None
            state: dict[str, Any] = dict(snap.values)
            if not role_review_due(state):
                log_event("role_review.skip", reason="throttle", thread_id=tid)
                return None
            user_text, agent_reply = _last_turn_pair(state)
            if not user_text or not agent_reply:
                log_event("role_review.skip", reason="empty_turn", thread_id=tid)
                return None

            from src.agent_runner import llm_compute_gate

            t0 = time.perf_counter()
            with llm_compute_gate(label="role_review"):
                outcome = review_turn(user_text=user_text, agent_reply=agent_reply)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if outcome is None:
                log_event(
                    "role_review.skip",
                    reason="parse_or_llm",
                    thread_id=tid,
                    elapsed_ms=round(elapsed_ms, 1),
                )
                return elapsed_ms

            if not outcome.role_failed or not outcome.gold_reply.strip():
                log_event(
                    "role_review.ok",
                    thread_id=tid,
                    elapsed_ms=round(elapsed_ms, 1),
                )
                return elapsed_ms

            pair_id = insert_gold_pair(
                user_text=user_text,
                rejected_reply=agent_reply,
                chosen_reply=outcome.gold_reply,
                critique=outcome.critique,
                thread_id=tid,
            )
            log_event(
                "role_review.gold",
                thread_id=tid,
                pair_id=pair_id,
                elapsed_ms=round(elapsed_ms, 1),
            )
            return elapsed_ms


def spawn_role_review(*, thread_id: str, pending: bool) -> None:
    """Fire-and-forget role review after reply."""
    if not pending:
        return
    tid = (thread_id or "").strip()
    if not tid:
        return

    spawn_deduped_daemon(
        job="role_review",
        thread_id=tid,
        target=lambda: run_role_review_for_thread(thread_id=tid),
        thread_name=f"deferred-role-review-{tid}",
    )


def schedule_role_review_background(*, thread_id: str) -> bool:
    """Start deferred role review when enabled and due."""
    tid = (thread_id or "").strip()
    if not tid or not settings.role_review:
        return False
    from src.agent_runner import get_graph_app, graph_invoke_config

    app = get_graph_app()
    config = graph_invoke_config(thread_id=tid)
    try:
        snap = app.get_state(config)
        state = dict(snap.values) if snap and snap.values else {}
    except Exception as exc:
        _log.warning("role_review schedule skipped thread_id=%s err=%s", tid, exc)
        return False
    if not state or not role_review_due(state):
        return False

    spawn_role_review(thread_id=tid, pending=True)
    return True


__all__ = [
    "role_review_due",
    "run_role_review_for_thread",
    "schedule_role_review_background",
    "spawn_role_review",
]
