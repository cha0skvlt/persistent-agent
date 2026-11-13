"""Shared LangGraph invoke path for REPL, FastAPI, and Telegram."""

from __future__ import annotations

import contextlib
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError

from src.agent_errors import format_agent_turn_error
from src.config import settings
from src.graph.summary_node import session_summary_due
from src.obs.deferred_job_tracker import (
    active_deferred_jobs,
    live_turn_overlaps_deferred,
)
from src.obs.logs import log_event
from src.obs.metrics import record_turn_duration
from src.obs.spans import traced_agent_turn
from src.obs.turn_footer import tools_used_in_turn
from src.obs.turn_timing import (
    get_turn_timings,
    reset_turn_timing,
    set_turn_concurrent_deferred,
)
from src.reply.contract import resolve_visible_reply
from src.reverie.background import is_reverie_background_pass
from src.reverie.cancel import ReverieCancelledError, request_reverie_cancel
from src.tools.web_prefetch import auto_tools_from_state

_log = logging.getLogger(__name__)

_graph_app: Any | None = None
_llm_turn_semaphore: threading.Semaphore | None = None
_llm_turn_semaphore_cap: int | None = None
_llm_turn_semaphore_guard = threading.Lock()


def _get_llm_turn_semaphore() -> threading.Semaphore:
    """Shared LLM gate: live blocks; background try-acquires (see traffic light)."""
    global _llm_turn_semaphore, _llm_turn_semaphore_cap
    cap = max(1, int(settings.llm_max_concurrent_turns))
    with _llm_turn_semaphore_guard:
        if _llm_turn_semaphore is None or _llm_turn_semaphore_cap != cap:
            _llm_turn_semaphore = threading.Semaphore(cap)
            _llm_turn_semaphore_cap = cap
        return _llm_turn_semaphore


@contextlib.contextmanager
def _release_on_exit(sem: threading.Semaphore):
    """Release an already-acquired semaphore on exit (background try-acquire path)."""
    try:
        yield
    finally:
        sem.release()


@contextlib.contextmanager
def llm_compute_gate(*, label: str = "deferred"):
    """Traffic light for post-turn LLM work (summary/reflect/role_review).

    Blocks on the same gate live turns hold, so deferred compute never contends
    with the next user turn on the Metal box. Latency-insensitive → wait, not skip.
    """
    sem = _get_llm_turn_semaphore()
    t0 = time.perf_counter()
    sem.acquire()
    waited = (time.perf_counter() - t0) * 1000.0
    if waited > 50.0:
        log_event("llm_gate.deferred_wait", label=label, waited_ms=round(waited, 1))
    try:
        yield
    finally:
        sem.release()


def reset_llm_turn_semaphore() -> None:
    """Clear cached semaphore (tests)."""
    global _llm_turn_semaphore, _llm_turn_semaphore_cap
    with _llm_turn_semaphore_guard:
        _llm_turn_semaphore = None
        _llm_turn_semaphore_cap = None


@dataclass(frozen=True)
class AgentTurnResult:
    reply: str
    tools_used: tuple[str, ...]
    elapsed_ms: float
    node_timings: dict[str, float] = field(default_factory=dict)
    summary_pending: bool = False
    reflect_pending: bool = False
    role_review_pending: bool = False
    thread_id: str = ""
    reply_is_fallback: bool = False
    """True when ``reply`` is synthesized (empty-turn dump, turn error text),
    not genuine model prose — background passes must not deliver it to USER."""


def get_graph_app():
    """Process-wide compiled graph (lazy init)."""
    global _graph_app
    if _graph_app is None:
        from src.graph.workflow import build_app

        _graph_app = build_app()
    return _graph_app


def reset_graph_app() -> None:
    """Clear cached graph (tests)."""
    global _graph_app
    _graph_app = None


def graph_invoke_config(
    *, thread_id: str, reply_channel: str | None = None
) -> dict[str, Any]:
    configurable: dict[str, Any] = {"thread_id": thread_id}
    channel = (reply_channel or "").strip().lower()
    if channel:
        configurable["reply_channel"] = channel
    return {
        "configurable": configurable,
        "recursion_limit": settings.recursion_limit,
    }


def infer_reply_channel(thread_id: str) -> str | None:
    prefix = (settings.telegram_thread_prefix or "telegram").strip() or "telegram"
    tid = (thread_id or "").strip()
    if tid.startswith(f"{prefix}_"):
        return "telegram"
    return None


def turn_channel_label(thread_id: str, reply_channel: str | None = None) -> str:
    """Metric/log channel: telegram | api | cli | repl."""
    ch = (reply_channel or "").strip().lower()
    if ch:
        return ch
    inferred = infer_reply_channel(thread_id)
    if inferred:
        return inferred
    tid = (thread_id or "").strip()
    if tid == (settings.cli_default_thread_id or "").strip():
        return "cli"
    if tid == (settings.api_default_thread_id or "").strip() or tid.startswith("api_"):
        return "api"
    return "repl"


def _checkpoint_state(app: Any, config: dict[str, Any]) -> dict[str, Any]:
    try:
        snap = app.get_state(config)
    except Exception as exc:
        _log.warning("checkpoint state read failed: %s", exc)
        return {}
    if not snap or not snap.values:
        return {}
    return dict(snap.values)


def _deferred_overlap_flag(thread_id: str, *, seen_before_lock: bool) -> bool:
    """True when a deferred job is in flight for this thread during the live turn."""
    tid = (thread_id or "").strip()
    if not tid:
        return False
    return bool(seen_before_lock or live_turn_overlaps_deferred(tid))


def _touch_agent_activity() -> None:
    """Best-effort upsert of last_turn_at for time-sense / activity."""
    try:
        from src.db.pool import get_pool

        with get_pool().connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO agent_activity (agent_id, last_turn_at) "
                    "VALUES (%s, now()) "
                    "ON CONFLICT (agent_id) DO UPDATE SET last_turn_at = now()",
                    (settings.default_agent_id,),
                )
            conn.commit()
    except Exception as exc:
        _log.warning("agent_activity upsert failed (non-fatal): %s", exc)


def _emit_turn_agent(
    *,
    thread_id: str,
    channel: str | None,
    text: str,
    fallback: bool = False,
) -> None:
    if is_reverie_background_pass():
        return
    log_event(
        "turn.agent",
        channel=turn_channel_label(thread_id, channel),
        thread_id=thread_id,
        text=text or "",
        fallback=bool(fallback),
    )


def _emit_turn_telemetry(
    *,
    thread_id: str,
    channel: str | None,
    elapsed_ms: float,
    tools: tuple[str, ...],
    concurrent_deferred_job: bool,
    summary_pending: bool = False,
    reflect_pending: bool = False,
    role_review_pending: bool = False,
    node_timings: dict[str, float] | None = None,
) -> None:
    log_event(
        "turn.complete",
        elapsed_ms=round(elapsed_ms, 1),
        tools=list(tools),
        thread_id=thread_id,
        channel=turn_channel_label(thread_id, channel),
        summary_pending=summary_pending,
        reflect_pending=reflect_pending,
        role_review_pending=role_review_pending,
        concurrent_deferred_job=concurrent_deferred_job,
        deferred_jobs=sorted(active_deferred_jobs(thread_id)),
        node_timings={
            str(k): round(float(v), 1)
            for k, v in (node_timings or {}).items()
            if k is not None
        },
    )
    record_turn_duration(
        elapsed_ms / 1000.0,
        turn_channel_label(thread_id, channel),
        concurrent_deferred_job=concurrent_deferred_job,
    )


def run_agent_turn_detailed(
    message: str,
    *,
    thread_id: str,
    session_uuid: str | None = None,
    diary_source: str | None = None,
) -> AgentTurnResult:
    """Run one user turn synchronously; return reply text and turn metadata."""
    text = (message or "").strip()
    if not text:
        return AgentTurnResult(reply="", tools_used=(), elapsed_ms=0.0)

    # Live turns preempt background reverie; a background pass must not
    # cancel its own token.
    if not is_reverie_background_pass():
        request_reverie_cancel()

    from src.graph.session_summary_runner import thread_summary_lock

    tid = (thread_id or "").strip()
    overlap_before_lock = live_turn_overlaps_deferred(tid)
    with thread_summary_lock(tid):
        concurrent_deferred = _deferred_overlap_flag(
            tid, seen_before_lock=overlap_before_lock
        )
        return _run_agent_turn_locked(
            text,
            thread_id=tid,
            session_uuid=session_uuid,
            concurrent_deferred_job=concurrent_deferred,
            diary_source=diary_source,
        )


def _run_agent_turn_locked(
    text: str,
    *,
    thread_id: str,
    session_uuid: str | None,
    concurrent_deferred_job: bool = False,
    diary_source: str | None = None,
) -> AgentTurnResult:
    app = get_graph_app()
    sid = (session_uuid or "").strip() or str(uuid.uuid4())
    reset_turn_timing()
    set_turn_concurrent_deferred(concurrent_deferred_job)
    t0 = time.perf_counter()
    channel = infer_reply_channel(thread_id)
    channel_label = turn_channel_label(thread_id, channel)
    if not is_reverie_background_pass():
        log_event(
            "turn.user",
            channel=channel_label,
            thread_id=thread_id,
            text=text,
        )
    invoke_config = graph_invoke_config(thread_id=thread_id, reply_channel=channel)
    try:
        with traced_agent_turn(
            thread_id=thread_id,
            channel=turn_channel_label(thread_id, channel),
            concurrent_deferred_job=concurrent_deferred_job,
        ):
            # Traffic light, red: one LLM gate for everyone. Live blocks on it
            # (background yields fast via cancel + stream abort); background
            # only try-acquires — never starts while a live turn runs.
            sem = _get_llm_turn_semaphore()
            if is_reverie_background_pass():
                if not sem.acquire(blocking=False):
                    raise ReverieCancelledError("live turn holds the LLM gate")
                turn_gate = _release_on_exit(sem)
            else:
                turn_gate = sem
            with turn_gate:
                invoke_payload: dict[str, Any] = {
                    "messages": [HumanMessage(content=text)],
                    "current_time": "",
                    "session_uuid": sid,
                }
                if diary_source:
                    invoke_payload["diary_source"] = diary_source
                result = app.invoke(
                    invoke_payload,
                    config=invoke_config,
                )
    except GraphRecursionError:
        limit = settings.recursion_limit
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        msg = f"Graph step limit ({limit}). Turn stopped — shorten or simplify the ask."
        if is_reverie_background_pass():
            # Never enqueue operator/debug text as proactive outreach.
            reply = ""
        elif channel == "telegram":
            reply = f"Operator notice: Turn error GraphRecursionError\n\n{msg}"
        else:
            reply = f"Operator notice: {msg}"
        _emit_turn_agent(
            thread_id=thread_id,
            channel=channel,
            text=reply,
            fallback=True,
        )
        _emit_turn_telemetry(
            thread_id=thread_id,
            channel=channel,
            elapsed_ms=elapsed_ms,
            tools=(),
            concurrent_deferred_job=concurrent_deferred_job,
            node_timings=get_turn_timings(),
        )
        return AgentTurnResult(
            reply=reply,
            tools_used=(),
            elapsed_ms=elapsed_ms,
            node_timings=get_turn_timings(),
            thread_id=thread_id,
            reply_is_fallback=True,
        )
    except ReverieCancelledError:
        # Background reverie pass preempted by a live turn — let the
        # reverie runner record "cancelled" instead of an error reply.
        raise
    except Exception as exc:
        _log.exception("agent turn failed: %s", exc)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        # Background: empty + fallback flag — Ollama/template errors must not
        # become solitary/intent proactive messages (seen as raw JSON in TG).
        reply = (
            ""
            if is_reverie_background_pass()
            else format_agent_turn_error(exc, channel=channel)
        )
        _emit_turn_agent(
            thread_id=thread_id,
            channel=channel,
            text=reply,
            fallback=True,
        )
        _emit_turn_telemetry(
            thread_id=thread_id,
            channel=channel,
            elapsed_ms=elapsed_ms,
            tools=(),
            concurrent_deferred_job=concurrent_deferred_job,
            node_timings=get_turn_timings(),
        )
        return AgentTurnResult(
            reply=reply,
            tools_used=(),
            elapsed_ms=elapsed_ms,
            node_timings=get_turn_timings(),
            thread_id=thread_id,
            reply_is_fallback=True,
        )

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    node_timings = get_turn_timings()
    msgs = result.get("messages") or []
    seen: set[str] = set()
    tools_list: list[str] = []
    for name in list(tools_used_in_turn(msgs)) + auto_tools_from_state(result):
        if name and name not in seen:
            seen.add(name)
            tools_list.append(name)
    tools = tuple(tools_list)
    visible = resolve_visible_reply(
        msgs,
        user_message=text,
        preserve_thinking=True,
        channel=channel,
        tools=tools,
    )
    reply = visible.text
    reply_is_fallback = visible.is_fallback
    # Telegram keeps ``<think>`` in reply for expandable blockquote; CLI/API/REPL
    # must never emit chain-of-thought — suppress the turn if nothing speakable remains.
    if (
        (channel or "").strip().lower() != "telegram"
        and reply
        and not reply_is_fallback
    ):
        from src.reply.visible import (
            promote_speech_from_thinking,
            split_redacted_thinking,
            strip_think_tag_leaks,
        )

        think, main, _trunc = split_redacted_thinking(reply)
        main = (main or "").strip()
        if not main and (think or "").strip():
            promoted = promote_speech_from_thinking(think, main)
            if promoted:
                main = (promoted[1] or "").strip()
            if not main:
                log_event(
                    "turn.think_only_reply_suppressed",
                    thread_id=thread_id,
                    chars=len(reply),
                )
        reply = strip_think_tag_leaks(main)

    checkpoint = _checkpoint_state(app, invoke_config)
    due = session_summary_due(checkpoint) if checkpoint else False
    summary_pending = bool(due and settings.summary_defer)
    if due and not settings.summary_defer:
        from src.graph.session_summary_runner import run_session_summary_if_due

        summary_ms = run_session_summary_if_due(thread_id=thread_id, lock_held=True)
        if summary_ms is not None:
            node_timings = dict(node_timings)
            node_timings["summarize_session"] = float(summary_ms)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

    from src.graph.reflect_runner import reflect_due

    reflect_is_due = bool(reflect_due(checkpoint)) if checkpoint else False
    reflect_pending = bool(reflect_is_due and settings.reflect_defer)

    has_reply = bool((reply or "").strip())
    review_due = False
    # Skip operator empty-turn dumps — gold pairs need genuine agent prose.
    if settings.role_review and has_reply and not reply_is_fallback and checkpoint:
        from src.graph.role_review_runner import role_review_due

        review_due = bool(role_review_due(checkpoint))
    role_review_pending = bool(review_due)

    concurrent_deferred_job = concurrent_deferred_job or live_turn_overlaps_deferred(
        thread_id
    )

    _emit_turn_agent(
        thread_id=thread_id,
        channel=channel,
        text=reply,
        fallback=reply_is_fallback,
    )
    _emit_turn_telemetry(
        thread_id=thread_id,
        channel=channel,
        elapsed_ms=elapsed_ms,
        tools=tools,
        concurrent_deferred_job=concurrent_deferred_job,
        summary_pending=summary_pending,
        reflect_pending=reflect_pending,
        role_review_pending=role_review_pending,
        node_timings=node_timings,
    )
    from src.mind.hub import record_agent_out, record_user_in

    if not is_reverie_background_pass():
        # Background passes are the agent's own activity — they must not mark
        # USER as active or land in the timeline as USER_IN.
        _touch_agent_activity()
        record_user_in(text=text, thread_id=thread_id, channel=channel)
    if not reply_is_fallback:
        # Operator empty-turn dumps must not land in mind hub as agent speech.
        record_agent_out(reply=reply, thread_id=thread_id, channel=channel)

    if is_reverie_background_pass() and reflect_pending and thread_id:
        from src.graph.reflect_runner import run_reflect_if_due

        reflect_ms = run_reflect_if_due(thread_id=thread_id, lock_held=True)
        if reflect_ms is not None:
            node_timings = dict(node_timings)
            node_timings["reflect_node"] = float(reflect_ms)

    return AgentTurnResult(
        reply=reply,
        tools_used=tools,
        elapsed_ms=elapsed_ms,
        node_timings=node_timings,
        summary_pending=summary_pending,
        reflect_pending=reflect_pending,
        role_review_pending=role_review_pending,
        thread_id=thread_id,
        reply_is_fallback=reply_is_fallback,
    )


def run_agent_turn(
    message: str,
    *,
    thread_id: str,
    session_uuid: str | None = None,
) -> str:
    """Run one user turn synchronously; return sanitized reply text."""
    return run_agent_turn_detailed(
        message,
        thread_id=thread_id,
        session_uuid=session_uuid,
    ).reply


async def arun_agent_turn(
    message: str,
    *,
    thread_id: str,
    session_uuid: str | None = None,
) -> AgentTurnResult:
    """Async wrapper: graph invoke runs in a worker thread."""
    import asyncio

    return await asyncio.to_thread(
        run_agent_turn_detailed,
        message,
        thread_id=thread_id,
        session_uuid=session_uuid,
    )
