"""Intent scheduler resilience — claim by run_at, fail backoff, terminal failed."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.config import settings
from src.reverie import intent_pass as ip


@pytest.fixture(autouse=True)
def _sched_settings(monkeypatch):
    monkeypatch.setattr(settings, "agent_tz", "UTC", raising=False)
    monkeypatch.setattr(settings, "default_agent_id", "agent", raising=False)
    monkeypatch.setattr(settings, "pulse_enabled", True, raising=False)


def _cleanup() -> None:
    from src.db.pool import get_pool

    aid = settings.default_agent_id
    with get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM intents WHERE agent_id = %s", (aid,))
            cur.execute("DELETE FROM mind_events WHERE agent_id = %s", (aid,))
        conn.commit()


def _insert_intent(
    *,
    status: str = "pending",
    run_at: datetime | None = None,
    attempts: int = 0,
    last_error: str | None = None,
    text: str = "scheduler resilience probe",
) -> str:
    from src.db.pool import get_pool

    when = run_at or (datetime.now(timezone.utc) - timedelta(minutes=1))
    with get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO intents
                  (agent_id, text, run_at, status, attempts, last_error)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id::text
                """,
                (
                    settings.default_agent_id,
                    text,
                    when,
                    status,
                    attempts,
                    last_error,
                ),
            )
            intent_id = cur.fetchone()[0]
        conn.commit()
    return intent_id


def _row(intent_id: str):
    from src.db.pool import get_pool

    with get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT status, attempts, last_error, run_at
                FROM intents WHERE id = %s::uuid
                """,
                (intent_id,),
            )
            return cur.fetchone()


@pytest.mark.integration
@pytest.mark.db
def test_claim_picks_pending_by_run_at(disposable_postgres):
    _cleanup()
    try:
        iid = _insert_intent(
            status="pending", run_at=datetime.now(timezone.utc) - timedelta(minutes=5)
        )
        claimed = ip._claim_due_intent()
        assert claimed is not None
        assert claimed[0] == iid
        assert _row(iid)[0] == "running"
    finally:
        _cleanup()


@pytest.mark.integration
@pytest.mark.db
def test_claim_skips_future_and_failed(disposable_postgres):
    _cleanup()
    try:
        _insert_intent(
            status="pending",
            run_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        _insert_intent(
            status="failed",
            run_at=datetime.now(timezone.utc) - timedelta(minutes=5),
        )
        assert ip._claim_due_intent() is None
    finally:
        _cleanup()


@pytest.mark.integration
@pytest.mark.db
def test_fail_backoff_and_terminal_failed(disposable_postgres, monkeypatch):
    _cleanup()
    notifies: list[str] = []

    def _notify(msg: str, *, alert_key: str = ""):
        notifies.append(msg)

    monkeypatch.setattr("src.heart.notify.notify_telegram", _notify)
    try:
        iid = _insert_intent(status="pending")
        ip._last_intent_error = "X"
        for _ in range(5):
            ip._park_failed_intent(iid)
        status, attempts, last_error, run_at = _row(iid)
        assert status == "failed"
        assert attempts == 5
        assert last_error == "X"
        assert run_at > datetime.now(timezone.utc)
        assert len(notifies) == 1
        assert "failed permanently" in notifies[0]
    finally:
        ip._last_intent_error = ""
        _cleanup()


@pytest.mark.integration
@pytest.mark.db
def test_success_resets_budget(disposable_postgres):
    _cleanup()
    try:
        iid = _insert_intent(
            status="running",
            attempts=3,
            last_error="prior",
        )
        ip._set_intent_status(iid, status="done", reset_attempts=True)
        status, attempts, last_error, _run_at = _row(iid)
        assert status == "done"
        assert attempts == 0
        assert last_error is None
    finally:
        _cleanup()


@pytest.mark.integration
@pytest.mark.db
def test_cancel_backoff_keeps_attempts(disposable_postgres, monkeypatch):
    from src.reverie.cancel import ReverieCancelledError

    _cleanup()
    try:
        iid = _insert_intent(status="pending", attempts=2)

        def _turn(*_a, **_k):
            raise ReverieCancelledError("live preempt")

        monkeypatch.setattr("src.agent_runner.run_agent_turn_detailed", _turn)
        assert ip.run_due_intent_if_any() is True
        status, attempts, _err, run_at = _row(iid)
        assert status == "pending"
        assert attempts == 2
        assert run_at > datetime.now(timezone.utc)
    finally:
        _cleanup()


@pytest.mark.integration
@pytest.mark.db
def test_list_intents_shows_failed(disposable_postgres):
    from src.tools.intent_tools import list_intents_impl

    _cleanup()
    try:
        from src.db.pool import get_pool

        iid = _insert_intent(status="failed", attempts=5, last_error="boom")
        with get_pool().connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE intents SET last_run_at = now() WHERE id = %s::uuid",
                    (iid,),
                )
            conn.commit()
        out = list_intents_impl()
        assert "failed/24h=" in out
    finally:
        _cleanup()
