"""Solitary cycle — agent's free background mind when not answering USER.

Solitary runs the SAME LangGraph as a live user turn.  The "user
message" is a freedom prompt: reflect on anything, use tools, curate
memory.  Writing to USER is her own decision via the message_user tool.
"""

from __future__ import annotations

import logging
import uuid

from src.config import settings
from src.obs.logs import log_event
from src.obs.metrics import record_reverie_pass
from src.pulse.rhythm import RhythmAction, mark_solitary_ran, may
from src.reverie.background_deliver import run_background_turn_and_deliver
from src.reverie.turn_kind import TurnKind

_log = logging.getLogger(__name__)

# Traffic light, yellow tail: a pass preempted by a live turn keeps its
# checkpoint thread; the next pass resumes that thought instead of starting
# a new one — no torn essays, no lost distillation.
_resume_thread_id: str | None = None

_RESUME_PROMPT = (
    "Your previous solitary pass was interrupted by a live USER turn. Above is your "
    "unfinished thought (context preserved). Finish it: complete what you started, "
    "distill the result into diary/memory, and close the pass. "
    "Do not start a new topic."
)


def _load_solitary_prompt() -> str:
    from src.reverie.prompts import build_solitary_user_message

    return build_solitary_user_message()


def _allow_solitary_fallback(reply: str, model_queued: bool) -> bool:
    """Solitary never auto-queues visible reply.

    Writing to USER is only via ``message_user``. Auto-fallback turned
    internal monologue ("not going to write") into Telegram spam.
    """
    return False


def run_solitary_pass_if_due() -> str:
    """Run one solitary reflection pass.

    Returns a status string: ran, skip_disabled, skip_interval,
    cancelled, error.
    """
    if not settings.solitary_enabled:
        return "skip_disabled"

    verdict = may(RhythmAction.SOLITARY_PASS)
    if not verdict:
        reason = verdict.reason or "skip"
        log_event("reverie.skip", reason=reason)
        record_reverie_pass(reason)
        return reason

    try:
        from src.policy_knobs import (
            apply_outcome_safe,
            solitary_deferred_for_live_busy,
        )

        if solitary_deferred_for_live_busy():
            log_event("reverie.skip", reason="skip_live_busy")
            record_reverie_pass("skip_live_busy")
            apply_outcome_safe("attention_deferred")
            return "skip_live_busy"
    except Exception as exc:
        _log.warning("live-busy defer check failed: %s", exc)

    from src.reverie.attention import begin_attention_clock, end_attention_clock
    from src.reverie.cancel import begin_reverie_cycle, end_reverie_cycle

    token = begin_reverie_cycle()
    clock_tok = begin_attention_clock()
    try:
        result = _do_solitary_pass(token)
        if result == "ran":
            mark_solitary_ran()
        log_event("reverie.run" if result == "ran" else "reverie.skip", reason=result)
        record_reverie_pass(result)
        return result
    finally:
        end_attention_clock(clock_tok)
        end_reverie_cycle(token)


def _thread_has_checkpoint(thread_id: str) -> bool:
    try:
        from src.agent_runner import get_graph_app, graph_invoke_config

        snap = get_graph_app().get_state(graph_invoke_config(thread_id=thread_id))
        return bool(snap and getattr(snap, "values", None))
    except Exception:
        return False


def _do_solitary_pass(token) -> str:
    """Run a full agent turn (with tools, tracing, footer) as solitary."""
    global _resume_thread_id
    if token.is_set():
        return "cancelled"

    from src.agent_runner import run_agent_turn_detailed

    if _resume_thread_id:
        thread_id = _resume_thread_id
        prompt = _RESUME_PROMPT
        log_event("reverie.resume", thread_id=thread_id)
    else:
        thread_id = f"solitary-{uuid.uuid4().hex[:8]}"
        prompt = _load_solitary_prompt()

    def _run():
        return run_agent_turn_detailed(
            prompt,
            thread_id=thread_id,
            session_uuid=f"solitary-{uuid.uuid4().hex[:8]}",
            diary_source="solitary",
        )

    delivered = run_background_turn_and_deliver(
        kind=TurnKind.SOLITARY,
        run_turn=_run,
        allow_fallback=_allow_solitary_fallback,
        fallback_trigger="solitary_say",
        on_cancelled="cancelled",
        on_error="error",
    )
    if delivered.result != "ran":
        # Preempted mid-thought: keep the thread for resume — but only when the
        # graph actually checkpointed something (gate-rejects leave no thought).
        if delivered.result == "cancelled" and _thread_has_checkpoint(thread_id):
            _resume_thread_id = thread_id
        return delivered.result

    _resume_thread_id = None
    turn = delivered.turn
    tools_used = list(getattr(turn, "tools_used", ()) or ())
    log_event(
        "reverie.run",
        tools=tools_used,
        elapsed_ms=getattr(turn, "elapsed_ms", 0),
        preview=(delivered.reply or "")[:80],
        tell_user=delivered.fallback_queued or delivered.model_queued,
        essay=True,
    )
    from src.policy_knobs import apply_outcome_safe

    # Close-loop: a completed hygiene lane burns pile-up debt.
    from src.reverie.prompts import last_solitary_lane_id

    # Mutating hygiene only — search alone must not burn debt (close-loop theater).
    _HYGIENE_MUTATE_TOOLS = frozenset(
        {
            "forget_memory",
            "memory_replace",
            "forget_diary",
            "diary_replace",
        }
    )
    if last_solitary_lane_id() == "hygiene" and _HYGIENE_MUTATE_TOOLS.intersection(
        tools_used
    ):
        apply_outcome_safe("hygiene_done")
    # Attention cost: unique tool names ≠ rounds — score multi-tool / wall-clock.
    from src.reverie.attention import solitary_used_attention_cost

    if solitary_used_attention_cost(
        tools_used=tools_used,
        elapsed_ms=float(getattr(turn, "elapsed_ms", 0) or 0),
    ):
        apply_outcome_safe("curiosity_spent")
    return "ran"


__all__ = ["run_solitary_pass_if_due"]
