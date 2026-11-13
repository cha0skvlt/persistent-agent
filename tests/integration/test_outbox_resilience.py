"""Outbox resilience — send-fail backoff, dead letter, queue cap, stale sending."""

from __future__ import annotations

import pytest

from src.config import settings
from src.heart import outreach as outreach_mod


@pytest.fixture(autouse=True)
def _outbox_settings(monkeypatch):
    monkeypatch.setattr(settings, "default_agent_id", "agent", raising=False)
    monkeypatch.setattr(settings, "outreach_enabled", True, raising=False)
    monkeypatch.setattr(settings, "telegram_owner_chat_id", "12345", raising=False)
    monkeypatch.setattr(settings, "telegram_bot_token", "tok", raising=False)
    monkeypatch.setattr(settings, "outreach_footer_grace_sec", 0.0, raising=False)


def _cleanup() -> None:
    from src.db.pool import get_pool

    aid = settings.default_agent_id
    with get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM outbox WHERE agent_id = %s", (aid,))
        conn.commit()


def _insert_outbox(
    *,
    status: str = "pending",
    draft: str = "outbox resilience probe",
    footer: str = "**→ turn 1s**",
    attempts: int = 0,
    next_attempt_at=None,
) -> str:
    from src.db.pool import get_pool

    with get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO outbox
                  (agent_id, trigger_type, draft_text, footer_text,
                   status, attempts, next_attempt_at)
                VALUES (%s, 'agent_say', %s, %s, %s, %s, %s)
                RETURNING id::text
                """,
                (
                    settings.default_agent_id,
                    draft,
                    footer,
                    status,
                    attempts,
                    next_attempt_at,
                ),
            )
            oid = cur.fetchone()[0]
        conn.commit()
    return oid


def _outbox_row(oid: str):
    from src.db.pool import get_pool

    with get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT status, attempts, next_attempt_at
                FROM outbox WHERE id = %s::uuid
                """,
                (oid,),
            )
            return cur.fetchone()


@pytest.mark.integration
@pytest.mark.db
def test_send_fail_backoff_then_dead(
    disposable_postgres, monkeypatch, reset_outreach_state
):
    _cleanup()
    monkeypatch.setattr(outreach_mod, "_send_telegram", lambda *_a, **_k: False)
    monkeypatch.setattr(outreach_mod, "_save_diary", lambda *_a, **_k: True)
    try:
        oid = _insert_outbox()
        from src.db.pool import get_pool

        with get_pool().connection() as conn:
            for expected_attempts in range(1, 6):
                outreach_mod._last_send_fail_at = 0.0
                if expected_attempts > 1:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            UPDATE outbox
                            SET next_attempt_at = now() - interval '1 minute'
                            WHERE id = %s::uuid
                            """,
                            (oid,),
                        )
                    conn.commit()
                outreach_mod.send_outreach_if_due(conn)
                conn.commit()
                status, attempts, next_at = _outbox_row(oid)
                assert attempts == expected_attempts
                if expected_attempts < 5:
                    assert status == "pending"
                    assert next_at is not None
                else:
                    assert status == "dead"
    finally:
        _cleanup()


@pytest.mark.integration
@pytest.mark.db
def test_claim_skips_backoff_row(disposable_postgres, reset_outreach_state):
    from datetime import datetime, timedelta, timezone

    _cleanup()
    try:
        _insert_outbox(next_attempt_at=datetime.now(timezone.utc) + timedelta(hours=1))
        from src.db.pool import get_pool

        with get_pool().connection() as conn:
            outreach_mod._recover_stale_sending(conn)
            assert outreach_mod._claim_pending_outbox(conn) is None
            conn.commit()
    finally:
        _cleanup()


@pytest.mark.integration
@pytest.mark.db
def test_queue_cap_rejects(disposable_postgres, monkeypatch):
    from src.reverie.turn_kind import TurnKind, turn_kind
    from src.tools import message_user as mu

    _cleanup()
    cap = max(1, int(getattr(settings, "outreach_queue_max", 8)))
    monkeypatch.setattr(settings, "outreach_queue_max", cap, raising=False)
    monkeypatch.setattr(
        "src.policy_knobs.outreach_initiative_allows",
        lambda: True,
    )
    try:
        for i in range(cap):
            _insert_outbox(draft=f"queued-{i}")
        with turn_kind(TurnKind.SOLITARY):
            out = mu.message_user_impl("one more")
        assert out.startswith("Not sent")
        assert "queue" in out.casefold()
    finally:
        _cleanup()


@pytest.mark.integration
@pytest.mark.db
def test_sending_expired_emits_event(
    disposable_postgres, monkeypatch, reset_outreach_state
):
    _cleanup()
    events: list[str] = []

    def _spy(event: str, **_fields):
        events.append(event)

    monkeypatch.setattr(outreach_mod, "log_event", _spy)
    try:
        oid = _insert_outbox(status="sending")
        from src.db.pool import get_pool

        with get_pool().connection() as conn:
            outreach_mod._recover_stale_sending(conn)
            conn.commit()
        assert "outreach.sending_expired" in events
        status, _attempts, _next = _outbox_row(oid)
        assert status == "expired"
    finally:
        _cleanup()
