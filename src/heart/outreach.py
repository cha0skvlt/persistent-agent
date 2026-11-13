"""Proactive outreach — send pending outbox messages from heart tick.

No quiet hours, no silence guards: whether and when to write is the
agent's own decision at queue time (message_user tool)."""

from __future__ import annotations

import logging
import time

import psycopg

from src.config import settings
from src.obs.logs import log_event

_log = logging.getLogger(__name__)

_last_send_fail_at: float = 0.0
_SEND_FAIL_COOLDOWN = 300.0
_OUTBOX_MAX_ATTEMPTS = 5

_warned_no_chat_id = False


def _post_send_message(token: str, payload: dict) -> tuple[bool, str]:
    """POST one sendMessage. Returns ``(ok, error_description)``."""
    from src.api.telegram_send import post_send_message_sync

    return post_send_message_sync(token, payload)


def _send_chunk(token: str, chat_id: str, html_text: str, plain_fallback: str) -> bool:
    """One sendMessage — shared Delivery path (same as live TelegramClient)."""
    from src.api.telegram_send import send_chunk_sync

    return send_chunk_sync(
        token, chat_id, html_text, plain_fallback, log_prefix="outreach"
    )


def _send_telegram(text: str, footer: str = "") -> bool:
    """Send to TELEGRAM_OWNER_CHAT_ID through the shared Delivery layer."""
    from src.api.telegram_send import send_body_sync
    from src.reply.visible import sanitize_outreach_text

    token = (settings.telegram_bot_token or "").strip()
    chat_id = str(settings.telegram_owner_chat_id or "").strip()
    if not token or not chat_id:
        _log.info('{"event":"outreach.send.skipped","reason":"no_token_or_owner_chat"}')
        return False

    body = sanitize_outreach_text(text)
    if not body:
        return False
    footer_s = (footer or "").strip()
    return send_body_sync(
        token, chat_id, body, footer_s, outreach=True, log_prefix="outreach"
    )


def _save_diary(conn: psycopg.Connection, draft: str) -> bool:
    """Persist the outreach diary entry through the single diary write path.

    Goes through ``SelfReflection.save`` so the injection guard, the near-duplicate
    dedup, and the signature policy are the same as for a live reflection. The
    heart's connection is unused on purpose: the outbox row is already marked
    ``sent`` before this call, so a diary failure must never re-pend the message.
    """
    _ = conn
    try:
        from src.memory import get_default_self_reflection
        from src.reply.visible import sanitize_outreach_text

        # Diary keeps only what USER actually reads — never think/tool dumps.
        visible = sanitize_outreach_text(draft)
        if not visible:
            return True
        content = f"I wrote first: {visible}"
        saved = get_default_self_reflection().save(
            content,
            source="outreach",
            allow_missing_embedding=True,
        )
        _log.info('{"event":"outreach.diary.saved","saved":%s}', saved)
        return True
    except Exception as exc:
        _log.warning('{"event":"outreach.diary.failed","detail":"%s"}', exc)
        return False


def _recover_stale_sending(conn: psycopg.Connection) -> None:
    """Expire crash-orphaned ``sending`` rows (never re-pend).

    Claim happens before Telegram so a crash after send must not redeliver.
    A ``sending`` row at tick start is ambiguous (sent vs not) — expire it.
    Explicit send failures still release ``sending``→``pending`` in-path.
    """
    cur = conn.execute(
        """
        UPDATE outbox
        SET status = 'expired'
        WHERE status = 'sending' AND agent_id = %s
        RETURNING id
        """,
        (settings.default_agent_id,),
    )
    fetched = cur.fetchall() if cur is not None else None
    rows = fetched if isinstance(fetched, (list, tuple)) else ()
    for row in rows:
        _log.warning(
            '{"event":"outreach.sending_expired","id":"%s"}',
            row[0],
        )
        log_event("outreach.sending_expired", outbox_id=str(row[0]))


def _footer_grace_sec() -> float:
    return max(0.0, float(settings.outreach_footer_grace_sec))


def _synthetic_turn_footer(*, elapsed_ms: float | None = None) -> str:
    """Footer when stamp never arrived — prefer queue age as duration.

    ``agent_say`` always comes from ``message_user``; never emit bare ``→ turn —``.
    """
    from src.obs.turn_footer import orphan_outreach_footer

    return orphan_outreach_footer(elapsed_ms=elapsed_ms)


def _claim_pending_outbox(
    conn: psycopg.Connection,
) -> tuple[str, str, str, float | None] | None:
    """Atomically claim oldest eligible pending row → ``sending``.

    Rows with empty ``footer_text`` younger than ``OUTREACH_FOOTER_GRACE_SEC``
    are skipped so solitary/intent can stamp ``→ turn`` after the graph ends.
    Older orphans are claimed and get a synthetic footer at send time.

    Returns ``(id, draft_text, footer_text, age_ms)``.
    """
    grace = _footer_grace_sec()
    row = conn.execute(
        """
        UPDATE outbox
        SET status = 'sending'
        WHERE id = (
            SELECT id FROM outbox
            WHERE status = 'pending' AND agent_id = %s
              AND (next_attempt_at IS NULL OR next_attempt_at <= now())
              AND (
                (footer_text IS NOT NULL AND btrim(footer_text) <> '')
                OR created_at <= now() - make_interval(secs => %s)
              )
            ORDER BY created_at
            LIMIT 1
            FOR UPDATE SKIP LOCKED
        )
        RETURNING id, draft_text, COALESCE(footer_text, ''),
                  EXTRACT(EPOCH FROM (now() - created_at)) * 1000.0
        """,
        (settings.default_agent_id, grace),
    ).fetchone()
    if row is None:
        return None
    age_ms: float | None
    try:
        age_ms = float(row[3]) if row[3] is not None else None
    except (TypeError, ValueError):
        age_ms = None
    return str(row[0]), str(row[1]), str(row[2] or ""), age_ms


def send_outreach_if_due(conn: psycopg.Connection) -> None:
    """Check outbox and send one pending message if all guards pass.

    Claim (``pending`` → ``sending``) happens *before* Telegram so a crash
    after send cannot redeliver the same row as still-pending.
    """
    global _last_send_fail_at, _warned_no_chat_id

    if not settings.outreach_enabled:
        return

    from src.agent_mode import is_asleep

    if is_asleep():
        return

    if not str(settings.telegram_owner_chat_id or "").strip():
        if not _warned_no_chat_id:
            _log.warning(
                '{"event":"outreach.skipped","reason":"telegram_owner_chat_id not configured"}'
            )
            _warned_no_chat_id = True
        return

    if not (settings.telegram_bot_token or "").strip():
        _log.info(
            '{"event":"outreach.skipped","reason":"telegram_bot_token not configured"}'
        )
        return

    if time.time() - _last_send_fail_at < _SEND_FAIL_COOLDOWN:
        _log.info('{"event":"outreach.skipped","reason":"send_fail_cooldown"}')
        return

    _recover_stale_sending(conn)
    claimed = _claim_pending_outbox(conn)
    if claimed is None:
        return

    outbox_id, draft, footer, age_ms = claimed
    footer_s = (footer or "").strip()
    if not footer_s:
        # Past grace — synthesize with queue age so duration is never blank.
        footer_s = _synthetic_turn_footer(elapsed_ms=age_ms)
        _log.info(
            '{"event":"outreach.footer_synthetic","id":"%s"}',
            outbox_id,
        )
    elif footer_s.startswith("**→ turn —**"):
        # Orphan stamp from older builds lacked duration — upgrade in place.
        footer_s = _synthetic_turn_footer(elapsed_ms=age_ms)
        _log.info(
            '{"event":"outreach.footer_duration_filled","id":"%s"}',
            outbox_id,
        )

    from src.agent_errors import looks_like_operator_error
    from src.reply.visible import sanitize_outreach_text

    clean = sanitize_outreach_text(draft)
    if (
        not clean
        or looks_like_operator_error(clean)
        or looks_like_operator_error(draft or "")
    ):
        # Poison draft (think-only / tool XML / turn-error dump) — drop.
        conn.execute(
            "UPDATE outbox SET status = 'sent', sent_at = now() WHERE id = %s",
            (outbox_id,),
        )
        reason = (
            "operator_error_prose"
            if looks_like_operator_error(clean or draft or "")
            else "empty_after_sanitize"
        )
        _log.info(
            '{"event":"outreach.dropped","id":"%s","reason":"%s"}',
            outbox_id,
            reason,
        )
        return

    if not _send_telegram(clean, footer_s):
        # Per-row retry budget: exponential backoff; poison rows die instead of
        # hammering forever. Global cooldown still throttles transport storms.
        row = conn.execute(
            """
            UPDATE outbox
            SET attempts = attempts + 1,
                next_attempt_at = now() + make_interval(
                  mins => LEAST(60, POW(2, attempts + 1))::int
                ),
                status = CASE
                  WHEN attempts + 1 >= %s THEN 'dead'
                  ELSE 'pending'
                END
            WHERE id = %s AND status = 'sending'
            RETURNING attempts, status
            """,
            (_OUTBOX_MAX_ATTEMPTS, outbox_id),
        ).fetchone()
        _last_send_fail_at = time.time()
        attempts = int(row[0]) if row else -1
        status = str(row[1]) if row else "?"
        if status == "dead":
            _log.warning(
                '{"event":"outreach.dead_letter","id":"%s","attempts":%s}',
                outbox_id,
                attempts,
            )
        else:
            _log.info(
                '{"event":"outreach.send_retry","id":"%s","attempts":%s}',
                outbox_id,
                attempts,
            )
        return

    # Mark sent *before* diary/hub side-effects — Telegram already delivered.
    conn.execute(
        "UPDATE outbox SET status = 'sent', sent_at = now() WHERE id = %s",
        (outbox_id,),
    )

    if not _save_diary(conn, clean):
        _log.warning(
            '{"event":"outreach.diary.failed_after_send","id":"%s"}', outbox_id
        )

    from src.mind.hub import record_outreach_sent

    record_outreach_sent(draft=clean)
    _log.info('{"event":"outreach.sent","id":"%s"}', outbox_id)


__all__ = ["send_outreach_if_due"]
