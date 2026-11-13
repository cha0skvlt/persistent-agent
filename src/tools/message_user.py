"""Proactive message to USER — the agent's own decision during background passes."""

from __future__ import annotations

import logging

from langchain_core.tools import tool

from src.config import settings
from src.obs.logs import log_event
from src.reverie.background import is_reverie_background_pass

_log = logging.getLogger(__name__)


def count_outbox(agent_id: str) -> int | None:
    """Total outbox rows for an agent, or ``None`` if the count failed.

    Callers that gate fallback delivery must treat ``None`` as fail-closed
    (assume the model may already have queued — do **not** auto-deliver).
    """
    try:
        from src.db.pool import get_pool

        with get_pool().connection() as conn:
            row = conn.execute(
                "SELECT count(*) FROM outbox WHERE agent_id = %s", (agent_id,)
            ).fetchone()
        return int(row[0]) if row else 0
    except Exception as exc:
        _log.warning("count_outbox failed: %s", exc)
        return None


def outbox_grew(before: int | None, after: int | None) -> bool:
    """True when outbox count increased, or when either count is unknown.

    Unknown counts fail closed: treat as «already queued» so solitary/intent
    fallback does not duplicate a message the model may have inserted.
    """
    if before is None or after is None:
        return True
    return after > before


def outbox_confirmed_grew(before: int | None, after: int | None) -> bool:
    """True only when both counts are known and ``after > before``.

    Used by outreach fulfillment checks: unknown counts must **not** skip the
    corrective/unfulfilled path (opposite of :func:`outbox_grew` fail-closed).
    """
    if before is None or after is None:
        return False
    return after > before


def message_user_impl(
    text: str,
    *,
    preserve_thinking: bool = False,
    footer: str = "",
) -> str:
    # preserve_thinking kept for call-site compat; outreach never stores think.
    _ = preserve_thinking
    from src.reply.visible import sanitize_outreach_text

    raw = (text or "").strip()
    text = sanitize_outreach_text(raw)
    footer_s = (footer or "").strip()
    from src.reply.visible import is_empty_claim

    if text and is_empty_claim(text):
        log_event("outreach.suppressed", reason="empty_claim")
        return (
            "Not sent: that was a void denial ('emptiness'), not a message. "
            "Call message_user again with a real spoken cut."
        )
    if raw and not text:
        # Had content, but only think / tool XML / scaffolding remained.
        reason = "thinking_only"
        low = raw.lower()
        if "<tool_call" in low or "<function=" in low or '"name"' in raw[:80]:
            reason = "tool_dump"
        log_event("outreach.suppressed", reason=reason)
        return (
            "Not sent: that was internal (reasoning or a tool call), not a "
            "message. Call message_user again with only the words meant for USER."
        )
    if not text:
        return (
            "Not sent: empty text — call again with the full message in "
            'the text argument, e.g. message_user(text="…").'
        )
    # Never queue Ollama/template dumps or turn-error prose as proactive mail.
    from src.agent_errors import looks_like_operator_error

    if looks_like_operator_error(text):
        log_event("outreach.suppressed", reason="operator_error_prose")
        return "Not sent: that was an internal error dump, not a message for USER."
    if not is_reverie_background_pass():
        return "Not sent: this is a live conversation — just answer directly."
    if not settings.outreach_enabled:
        log_event("outreach.suppressed", reason="outreach_disabled")
        return "Not sent: OUTREACH_ENABLED=0 — proactive delivery is off."
    try:
        from src.policy_knobs import outreach_initiative_allows

        if not outreach_initiative_allows():
            log_event("outreach.suppressed", reason="low_initiative")
            return (
                "Not sent: outreach_initiative too low — "
                "behavior prior refused proactive send."
            )
    except Exception as exc:
        _log.warning("outreach initiative check failed: %s", exc)
        log_event("outreach.suppressed", reason="initiative_check_failed")
        return (
            "Not sent: outreach initiative check failed — "
            "behavior prior unavailable, refuse send."
        )
    try:
        from src.db.pool import get_pool

        cap = max(1, int(getattr(settings, "outreach_queue_max", 8)))
        with get_pool().connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT count(*) FROM outbox "
                    "WHERE agent_id = %s AND status = 'pending'",
                    (settings.default_agent_id,),
                )
                row = cur.fetchone()
                if row and int(row[0]) >= cap:
                    log_event("outreach.suppressed", reason="queue_full", cap=cap)
                    return (
                        "Not sent: the outgoing queue is full — wait for the "
                        "messages already queued to be delivered, do not duplicate."
                    )
                cur.execute(
                    "INSERT INTO outbox "
                    "(agent_id, trigger_type, draft_text, footer_text) "
                    "VALUES (%s, %s, %s, %s)",
                    (
                        settings.default_agent_id,
                        "agent_say",
                        text,
                        footer_s or None,
                    ),
                )
            conn.commit()
    except Exception as exc:
        _log.warning("message_user outbox insert failed: %s", exc)
        return "Not sent: outbox write failed, try again later."

    from src.mind.hub import record_outreach_queued
    from src.policy_knobs import apply_outcome_safe

    record_outreach_queued(draft=text)
    apply_outcome_safe("outreach_fulfilled")
    log_event("outreach.queued", trigger="agent_say", preview=text[:60])
    return "Queued for delivery to USER."


def stamp_pending_outbox_footer(agent_id: str, footer: str) -> int:
    """Attach the graph-turn footer to pending outbox rows still missing one.

    ``message_user`` may queue mid-turn (footer unknown yet). After the pass,
    stamp every pending row without a footer so heart sends the same
    ``→ turn …`` line as live Telegram replies.
    """
    footer_s = (footer or "").strip()
    if not footer_s:
        return 0
    try:
        from src.db.pool import get_pool

        with get_pool().connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE outbox
                    SET footer_text = %s
                    WHERE id IN (
                      SELECT id FROM outbox
                      WHERE agent_id = %s
                        AND status = 'pending'
                        AND (footer_text IS NULL OR btrim(footer_text) = '')
                        AND created_at > now() - interval '30 minutes'
                      ORDER BY created_at DESC
                      LIMIT 8
                    )
                    """,
                    (footer_s, agent_id),
                )
                n = int(cur.rowcount or 0)
            conn.commit()
        if n:
            log_event("outreach.footer_stamped", count=n, preview=footer_s[:60])
        return n
    except Exception as exc:
        _log.warning("stamp_pending_outbox_footer failed: %s", exc)
        return 0


@tool
def message_user(text: str = "") -> str:
    """Write to USER on your own initiative (background passes only).

    REQUIRED: put the full message for USER in the ``text`` argument —
    that string is what gets queued for Telegram. Use when you have
    something to say — thought, follow-up, question, report you promised.
    No-op when OUTREACH_ENABLED=0 (returns Not sent).
    """
    return message_user_impl(text)


__all__ = [
    "message_user",
    "message_user_impl",
    "count_outbox",
    "outbox_grew",
    "outbox_confirmed_grew",
    "stamp_pending_outbox_footer",
]
