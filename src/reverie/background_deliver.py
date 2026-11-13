"""BackgroundTurn orchestrator — shared solitary/intent outbox tail."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from src.config import settings
from src.obs.logs import log_event
from src.reverie.turn_kind import TurnKind, turn_kind

_log = logging.getLogger(__name__)


@dataclass
class BackgroundDeliverResult:
    """Outcome of a background graph pass + outbox delivery tail."""

    result: str  # ran / done / cancelled / failed / error
    reply: str
    turn: Any | None
    model_queued: bool
    fallback_queued: bool


def _queue_fallback(
    reply: str,
    *,
    kind: TurnKind,
    trigger: str,
) -> bool:
    from src.agent_errors import looks_like_operator_error
    from src.tools.message_user import message_user_impl

    text = (reply or "").strip()
    if not text or looks_like_operator_error(text):
        if text and looks_like_operator_error(text):
            log_event(
                "outreach.suppressed",
                reason="operator_error_prose",
                preview=text[:80],
            )
        return False
    with turn_kind(kind):
        result = message_user_impl(text)
    if result.startswith("Queued"):
        log_event("outreach.queued", trigger=trigger, preview=text[:60])
        return True
    if kind is TurnKind.INTENT:
        log_event("intent.no_message", preview=text[:60])
    return False


def apply_outbox_delivery_tail(
    *,
    kind: TurnKind,
    turn: Any,
    outbox_before: int | None,
    agent_id: str | None = None,
    allow_fallback: Callable[[str, bool], bool] | None = None,
    fallback_deliver: Callable[[str], bool] | None = None,
    fallback_trigger: str = "background_say",
) -> BackgroundDeliverResult:
    """Shared outbox tail: grew → optional fallback → stamp footer.

    Used by ``run_background_turn_and_deliver`` and by callers that already
    ran the graph (e.g. intent tests that patch ``_run_intent_graph``).
    """
    from src.agent_errors import looks_like_operator_error
    from src.obs.turn_footer import format_turn_log_footer
    from src.tools.message_user import (
        count_outbox,
        outbox_grew,
        stamp_pending_outbox_footer,
    )

    aid = agent_id or settings.default_agent_id
    model_queued = outbox_grew(outbox_before, count_outbox(aid))
    raw_reply = (getattr(turn, "reply", None) or "").strip()
    stripped_fallback = bool(getattr(turn, "reply_is_fallback", False))
    stripped_operator = bool(raw_reply) and looks_like_operator_error(raw_reply)
    reply = "" if stripped_fallback or stripped_operator else raw_reply
    if (stripped_fallback or stripped_operator) and not model_queued:
        # Silence is not success — operator dump must not masquerade as agent speech.
        log_event(
            "background.silence",
            kind=kind.value,
            reason=(
                "synthetic_fallback" if stripped_fallback else "operator_error_prose"
            ),
            preview=(raw_reply or "")[:80],
        )

    fallback_queued = False
    may_fallback = True
    if allow_fallback is not None:
        may_fallback = allow_fallback(reply, model_queued)
    elif model_queued or not reply:
        may_fallback = False

    if may_fallback and not model_queued:
        if fallback_deliver is not None:
            fallback_queued = bool(fallback_deliver(reply))
        else:
            fallback_queued = _queue_fallback(
                reply, kind=kind, trigger=fallback_trigger
            )

    if turn is not None:
        tools = list(getattr(turn, "tools_used", ()) or ())
        # message_user may have queued even when tools_used was lost (error path).
        if (model_queued or fallback_queued) and "message_user" not in tools:
            tools.append("message_user")
        timings = getattr(turn, "node_timings", None)
        stamp_pending_outbox_footer(
            aid,
            format_turn_log_footer(
                elapsed_ms=float(getattr(turn, "elapsed_ms", 0.0) or 0.0),
                tools=tools,
                node_timings=timings if isinstance(timings, dict) else None,
            ),
        )

    status = "done" if kind is TurnKind.INTENT else "ran"
    return BackgroundDeliverResult(
        result=status,
        reply=reply,
        turn=turn,
        model_queued=model_queued,
        fallback_queued=fallback_queued,
    )


def _stamp_orphan_outbox_if_queued(
    agent_id: str,
    outbox_before: int | None,
    *,
    elapsed_ms: float,
) -> None:
    """If message_user queued before cancel/crash, stamp footer with real duration."""
    from src.obs.turn_footer import orphan_outreach_footer
    from src.tools.message_user import (
        count_outbox,
        outbox_confirmed_grew,
        stamp_pending_outbox_footer,
    )

    if not outbox_confirmed_grew(outbox_before, count_outbox(agent_id)):
        return
    n = stamp_pending_outbox_footer(
        agent_id,
        orphan_outreach_footer(elapsed_ms=max(0.0, float(elapsed_ms))),
    )
    if n:
        log_event(
            "outreach.footer_orphan_stamped",
            count=n,
            elapsed_ms=round(float(elapsed_ms)),
        )


def run_background_turn_and_deliver(
    *,
    kind: TurnKind,
    run_turn: Callable[[], Any],
    agent_id: str | None = None,
    allow_fallback: Callable[[str, bool], bool] | None = None,
    fallback_deliver: Callable[[str], bool] | None = None,
    fallback_trigger: str = "background_say",
    on_cancelled: str = "cancelled",
    on_error: str = "error",
) -> BackgroundDeliverResult:
    """Run a background graph turn, then apply the shared outbox delivery tail.

    Tail: count outbox → run → fail-closed grew → optional fallback → stamp footer.
    *run_turn* must raise ``ReverieCancelledError`` on preemption and return an
    ``AgentTurnResult``-like object with ``reply``, ``reply_is_fallback``.
    """
    import time

    from src.reverie.cancel import ReverieCancelledError
    from src.tools.message_user import count_outbox

    aid = agent_id or settings.default_agent_id
    outbox_before = count_outbox(aid)
    turn = None
    t0 = time.perf_counter()
    try:
        with turn_kind(kind):
            turn = run_turn()
    except ReverieCancelledError:
        # Live preemption after mid-turn message_user — stamp with wall-clock
        # duration so Telegram never shows bare ``→ turn —``.
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        _stamp_orphan_outbox_if_queued(aid, outbox_before, elapsed_ms=elapsed_ms)
        return BackgroundDeliverResult(
            result=on_cancelled,
            reply="",
            turn=None,
            model_queued=False,
            fallback_queued=False,
        )
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        _log.warning("background turn failed (%s): %s", kind.value, exc)
        log_event("reverie.error", detail=str(exc)[:200], kind=kind.value)
        _stamp_orphan_outbox_if_queued(aid, outbox_before, elapsed_ms=elapsed_ms)
        return BackgroundDeliverResult(
            result=on_error,
            reply="",
            turn=None,
            model_queued=False,
            fallback_queued=False,
        )

    return apply_outbox_delivery_tail(
        kind=kind,
        turn=turn,
        outbox_before=outbox_before,
        agent_id=aid,
        allow_fallback=allow_fallback,
        fallback_deliver=fallback_deliver,
        fallback_trigger=fallback_trigger,
    )


__all__ = [
    "BackgroundDeliverResult",
    "apply_outbox_delivery_tail",
    "run_background_turn_and_deliver",
]
