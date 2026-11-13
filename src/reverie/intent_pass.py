"""Execute due Pulse intents via the full LangGraph (outreach fulfillment)."""

from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from src.config import settings
from src.obs.logs import log_event
from src.obs.metrics import record_pulse_intent_run
from src.pulse.when_parser import next_run_at
from src.reverie.background_deliver import apply_outbox_delivery_tail
from src.reverie.turn_kind import TurnKind, turn_kind
from src.tools.message_user import count_outbox, outbox_confirmed_grew

_log = logging.getLogger(__name__)


def _model_queued_outbox(before: int | None, turn) -> bool:
    """True when outbox grew, or message_user ran (count may be unknown)."""
    after = count_outbox(settings.default_agent_id)
    if outbox_confirmed_grew(before, after):
        return True
    tools = getattr(turn, "tools_used", None) or ()
    return "message_user" in tools


# outreach gate: whole-word write/tell verbs (avoid "say" inside "essay").
_OUTREACH_VERB_RE = re.compile(
    r"(?iu)(?<!\w)(?:" r"write|tell|send|message|notify|remind|ping|report" r")(?!\w)"
)
# "report back / let me know" already implies a report to USER.
_OUTREACH_REPORT_RE = re.compile(
    r"(?iu)(?<!\w)(?:report(?:\s+back)?|let\s+(?:me|the\s+user)\s+know)(?!\w)"
)

_OUTREACH_CORRECTIVE_PROMPT = (
    "The intent requires writing to USER. In the previous pass message_user was not "
    "called successfully. Call message_user now with the finished text for USER. "
    "The visible reply in this pass is not delivered anywhere — only message_user."
)


def _tz() -> ZoneInfo:
    return ZoneInfo(settings.agent_tz)


def is_outreach_shaped(text: str) -> bool:
    """True when intent text asks to write/tell/report to USER (outreach gate).

    Verb alone is enough: schedule_intent often stores "write the results of X"
    without "me"/USER, and those still require message_user.
    """
    raw = text or ""
    if _OUTREACH_REPORT_RE.search(raw):
        return True
    return bool(_OUTREACH_VERB_RE.search(raw))


def _fmt_dt(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_tz())
    return dt.astimezone(_tz()).strftime("%Y-%m-%d %H:%M")


def _late_delta_line(*, now: datetime, run_at: datetime) -> str | None:
    if now.tzinfo is None:
        now = now.replace(tzinfo=_tz())
    if run_at.tzinfo is None:
        run_at = run_at.replace(tzinfo=_tz())
    seconds = (now - run_at).total_seconds()
    if seconds <= 3600:
        return None
    hours = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    if hours > 0 and mins > 0:
        delta = f"{hours}h {mins}m"
    elif hours > 0:
        delta = f"{hours}h"
    else:
        delta = f"{mins}m"
    return f"The scheduled time passed {delta} ago — take that into account."


def _build_intent_prompt(*, text: str, run_at: datetime) -> str:
    now = datetime.now(_tz())
    lines = [
        f'Now {_fmt_dt(now)}. You scheduled this for yourself at {_fmt_dt(run_at)}: "{text}".',
        "This is your own turn — do it: think, check memory or the web if needed.",
        "Important: this is a background pass — your visible reply is NOT delivered "
        "anywhere. The only way to say something to USER is to call message_user with "
        "the finished text. If the intent implies writing to USER — finish the pass by "
        "calling message_user, otherwise the intent failed.",
        "run_selftest is available in this pass — if the intent requires a check, "
        "call it, then write the result to USER via message_user.",
    ]
    late = _late_delta_line(now=now, run_at=run_at)
    if late:
        lines.append(late)
    return "\n".join(lines)


def _claim_due_intent() -> tuple[str, str, datetime, str | None] | None:
    from src.db.pool import get_pool

    aid = settings.default_agent_id
    with get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                WITH picked AS (
                  SELECT id FROM intents
                  WHERE agent_id = %s
                    AND status IN ('pending', 'due')
                    AND run_at <= now()
                  ORDER BY run_at
                  LIMIT 1
                  FOR UPDATE SKIP LOCKED
                )
                UPDATE intents
                SET status = 'running', last_run_at = now()
                WHERE id IN (SELECT id FROM picked)
                  AND status IN ('pending', 'due')
                RETURNING id::text, text, run_at, recur_rule
                """,
                (aid,),
            )
            row = cur.fetchone()
        conn.commit()
    if not row:
        return None
    return str(row[0]), str(row[1]), row[2], row[3]


def _set_intent_status(
    intent_id: str,
    *,
    status: str,
    run_at: datetime | None = None,
    reset_attempts: bool = False,
) -> None:
    from src.db.pool import get_pool

    reset_sql = ", attempts = 0, last_error = NULL" if reset_attempts else ""
    with get_pool().connection() as conn:
        with conn.cursor() as cur:
            if run_at is not None:
                cur.execute(
                    f"UPDATE intents SET status = %s, run_at = %s{reset_sql} "
                    "WHERE id = %s::uuid",
                    (status, run_at, intent_id),
                )
            else:
                cur.execute(
                    f"UPDATE intents SET status = %s{reset_sql} WHERE id = %s::uuid",
                    (status, intent_id),
                )
        conn.commit()


_INTENT_MAX_ATTEMPTS = 5
# Set by _run_intent_graph on failure; consumed by _park_failed_intent
# (reverie loop is single-threaded).
_last_intent_error: str = ""


def _park_failed_intent(intent_id: str) -> None:
    """Retry budget: exponential backoff (2^n min, cap 60); attempts>=max → terminal
    'failed' + owner notice. Kills the infinite fail→due→fail hammer."""
    from src.db.pool import get_pool

    err = (_last_intent_error or "intent graph run failed").strip()[:500]
    with get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE intents
                SET attempts = attempts + 1,
                    last_error = %s,
                    status = CASE
                      WHEN attempts + 1 >= %s THEN 'failed'
                      ELSE 'pending'
                    END,
                    run_at = now() + make_interval(
                      mins => LEAST(60, POW(2, attempts + 1))::int
                    )
                WHERE id = %s::uuid
                RETURNING attempts, status
                """,
                (err, _INTENT_MAX_ATTEMPTS, intent_id),
            )
            row = cur.fetchone()
        conn.commit()
    attempts = int(row[0]) if row else -1
    status = str(row[1]) if row else "?"
    log_event(
        "intent.fail_backoff",
        intent_id=intent_id,
        attempts=attempts,
        status=status,
        err=err[:120],
    )
    if status == "failed":
        try:
            from src.heart.notify import notify_telegram

            notify_telegram(
                f"🔴 intent failed permanently ({attempts} attempts): {err[:160]}",
                alert_key="intent_failed",
            )
        except Exception as exc:
            _log.warning("intent failed-notify skipped: %s", exc)


def _run_intent_graph(
    token,
    *,
    intent_id: str,
    text: str,
    run_at: datetime,
    thread_id: str | None = None,
    corrective: bool = False,
) -> tuple[str, str, object | None, str]:
    """Run graph pass only. Returns ``(result, reply, turn, thread_id)``.

    Outbox delivery is applied by ``run_due_intent_if_any`` via BackgroundTurn.
    """
    tid = (thread_id or "").strip() or f"intent-{uuid.uuid4().hex[:8]}"
    if token.is_set():
        return "cancelled", "", None, tid

    from src.reverie.cancel import ReverieCancelledError

    try:
        from src.agent_runner import run_agent_turn_detailed

        prompt = (
            _OUTREACH_CORRECTIVE_PROMPT
            if corrective
            else _build_intent_prompt(text=text, run_at=run_at)
        )

        with turn_kind(TurnKind.INTENT):
            turn = run_agent_turn_detailed(
                prompt,
                thread_id=tid,
                session_uuid=f"intent-{uuid.uuid4().hex[:8]}",
                diary_source="intent",
            )
        if turn.reply_is_fallback:
            return "done", "", turn, tid
        return "done", (turn.reply or "").strip(), turn, tid
    except ReverieCancelledError:
        return "cancelled", "", None, tid
    except Exception as exc:
        _log.warning("intent graph invoke failed: %s", exc)
        log_event("intent.run_failed", intent_id=intent_id, detail=str(exc)[:200])
        global _last_intent_error
        _last_intent_error = f"{type(exc).__name__}: {exc}"
        return "failed", "", None, tid


def _finish_intent_status(
    intent_id: str,
    *,
    recur_rule: str | None,
    now: datetime,
) -> None:
    if recur_rule:
        _set_intent_status(
            intent_id,
            status="pending",
            run_at=next_run_at(recur_rule, now),
            reset_attempts=True,
        )
    else:
        _set_intent_status(intent_id, status="done", reset_attempts=True)


def run_due_intent_if_any() -> bool:
    """Claim one due intent and run it through LangGraph. Returns True if claimed."""
    if not settings.pulse_enabled:
        return False

    claimed = _claim_due_intent()
    if claimed is None:
        return False

    intent_id, text, run_at, recur_rule = claimed

    from src.reverie.cancel import begin_reverie_cycle, end_reverie_cycle

    aid = settings.default_agent_id
    outbox_before = count_outbox(aid)
    outreach = is_outreach_shaped(text)

    token = begin_reverie_cycle()
    result = "failed"
    try:
        result, reply, turn, thread_id = _run_intent_graph(
            token,
            intent_id=intent_id,
            text=text,
            run_at=run_at,
        )
        now = datetime.now(_tz())

        if result == "done":
            # outreach gate: no message_user → one corrective nudge.
            # Prefer outbox growth; also trust tools_used (count can be None).
            model_queued = _model_queued_outbox(outbox_before, turn)
            if outreach and not model_queued:
                result, reply, turn, thread_id = _run_intent_graph(
                    token,
                    intent_id=intent_id,
                    text=text,
                    run_at=run_at,
                    thread_id=thread_id,
                    corrective=True,
                )
                if result != "done":
                    # Same as first-pass cancel: never snap back to due (reclaim thrash).
                    if result == "cancelled":
                        backoff_at = datetime.now(_tz()) + timedelta(minutes=2)
                        _set_intent_status(
                            intent_id, status="pending", run_at=backoff_at
                        )
                        log_event(
                            "intent.cancel_backoff",
                            intent_id=intent_id,
                            run_at=_fmt_dt(backoff_at),
                        )
                    else:
                        _park_failed_intent(intent_id)
                    log_event("intent.run", result=result, intent_id=intent_id)
                    record_pulse_intent_run(result)
                    return True
                model_queued = _model_queued_outbox(outbox_before, turn)
                if not model_queued:
                    # Second failure: do not invent outreach text in code.
                    log_event("intent.outreach_unfulfilled", intent_id=intent_id)
                    from src.policy_knobs import apply_outcome_safe

                    apply_outcome_safe("outreach_unfulfilled")
                    _set_intent_status(intent_id, status="done")
                    from src.mind.hub import record_intent_run

                    record_intent_run(text=text)
                    log_event("intent.run", result="unfulfilled", intent_id=intent_id)
                    record_pulse_intent_run("unfulfilled")
                    return True

            # Shared BackgroundTurn tail. For outreach-shaped intents the model
            # (or corrective round) must have queued — never invent body in code.
            def allow_fallback(r: str, mq: bool) -> bool:
                return (not outreach) and bool(r) and not mq

            if turn is not None:
                apply_outbox_delivery_tail(
                    kind=TurnKind.INTENT,
                    turn=turn,
                    outbox_before=outbox_before,
                    agent_id=aid,
                    allow_fallback=allow_fallback,
                    fallback_trigger="intent_say",
                )
            elif reply:
                from types import SimpleNamespace

                apply_outbox_delivery_tail(
                    kind=TurnKind.INTENT,
                    turn=SimpleNamespace(reply=reply, reply_is_fallback=False),
                    outbox_before=outbox_before,
                    agent_id=aid,
                    allow_fallback=allow_fallback,
                    fallback_trigger="intent_say",
                )
            _finish_intent_status(intent_id, recur_rule=recur_rule, now=now)
            from src.mind.hub import record_intent_run

            record_intent_run(text=text)
        elif result == "cancelled":
            # Live preempt: if message_user already queued, treat as fulfilled —
            # re-parking would re-run and duplicate Telegram outreach.
            if _model_queued_outbox(outbox_before, turn):
                from src.reverie.background_deliver import (
                    _stamp_orphan_outbox_if_queued,
                )

                elapsed = (
                    float(getattr(turn, "elapsed_ms", 0.0) or 0.0) if turn else 0.0
                )
                _stamp_orphan_outbox_if_queued(aid, outbox_before, elapsed_ms=elapsed)
                _finish_intent_status(intent_id, recur_rule=recur_rule, now=now)
                from src.mind.hub import record_intent_run

                record_intent_run(text=text)
                log_event(
                    "intent.cancel_fulfilled",
                    intent_id=intent_id,
                    reason="outbox_grew",
                )
            else:
                # Immediate status=due causes reclaim thrash — park briefly.
                backoff_at = datetime.now(_tz()) + timedelta(minutes=2)
                _set_intent_status(intent_id, status="pending", run_at=backoff_at)
                log_event(
                    "intent.cancel_backoff",
                    intent_id=intent_id,
                    run_at=_fmt_dt(backoff_at),
                )
        else:
            _park_failed_intent(intent_id)
        log_event("intent.run", result=result, intent_id=intent_id)
        record_pulse_intent_run(result)
        return True
    finally:
        end_reverie_cycle(token)


__all__ = ["is_outreach_shaped", "run_due_intent_if_any"]
