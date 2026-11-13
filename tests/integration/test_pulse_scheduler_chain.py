"""Pulse scheduler chain — schedule → due → run → outbox → (mocked) Telegram.

Deterministic: LLM / Telegram mocked; real Postgres + real HTC / outreach gate / hub / heart.
Requires ``TEST_DATABASE_URL`` (never production ``DATABASE_URL``).
"""

from __future__ import annotations

import uuid
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage

from src.config import settings
from src.db.pool import get_pool
from src.heart import outreach as outreach_mod
from src.heart.pulse import mark_due_intents
from src.reverie import intent_pass as intent_pass_mod
from src.reverie.turn_kind import intent_background_pass
from src.tools.intent_tools import schedule_intent_impl


@pytest.fixture(autouse=True)
def _pulse_chain_settings(monkeypatch):
    monkeypatch.setattr(settings, "agent_tz", "UTC", raising=False)
    monkeypatch.setattr(settings, "default_agent_id", "agent", raising=False)
    monkeypatch.setattr(settings, "pulse_enabled", True, raising=False)
    monkeypatch.setattr(settings, "mind_hub_enabled", True, raising=False)
    monkeypatch.setattr(settings, "outreach_enabled", True, raising=False)
    monkeypatch.setattr(settings, "intents_max_pending", 50, raising=False)
    monkeypatch.setattr(settings, "intents_max_per_day", 50, raising=False)


def _cleanup_agent_rows() -> None:
    aid = settings.default_agent_id
    with get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM outbox WHERE agent_id = %s", (aid,))
            cur.execute("DELETE FROM intents WHERE agent_id = %s", (aid,))
            cur.execute("DELETE FROM mind_events WHERE agent_id = %s", (aid,))
        conn.commit()


def _outbox_drafts() -> list[str]:
    with get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT draft_text FROM outbox WHERE agent_id = %s ORDER BY created_at",
                (settings.default_agent_id,),
            )
            return [r[0] for r in cur.fetchall()]


def _mind_kinds() -> list[str]:
    with get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT kind FROM mind_events WHERE agent_id = %s ORDER BY created_at",
                (settings.default_agent_id,),
            )
            return [r[0] for r in cur.fetchall()]


def _intent_status(intent_id: str) -> str:
    with get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT status FROM intents WHERE id = %s::uuid",
                (intent_id,),
            )
            row = cur.fetchone()
    assert row is not None
    return row[0]


@pytest.mark.integration
@pytest.mark.db
def test_htc_qwen_xml_message_user_lands_in_outbox(
    disposable_postgres, monkeypatch, retrieve_mocks
):
    """Incident shape: Qwen XML in content → HTC → ToolNode → real outbox."""
    import src.graph.workflow as w
    import src.llm_factory as lf
    from tests.support.graph_fixtures import (
        fresh_memory_graph,
        fresh_thread_config,
        graph_test_settings,
        initial_turn_input,
    )

    _ = retrieve_mocks
    _cleanup_agent_rows()
    graph_test_settings(monkeypatch)
    monkeypatch.setattr(w.settings, "llm_supports_tools", False, raising=False)
    monkeypatch.setattr(lf.get_settings(), "llm_supports_tools", False, raising=False)
    lf.clear_llm_cache()

    report = f"Intent completed on time. Selftest ok. [{uuid.uuid4().hex[:8]}]"
    htc_body = (
        "<think>\nreport via message_user\n</think>\n"
        "<tool_call>\n"
        "<function=message_user>\n"
        f"<parameter=text>\n{report}\n</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    n = {"i": 0}

    def fake_invoke(*_a, **_k):
        n["i"] += 1
        if n["i"] == 1:
            return AIMessage(content=htc_body)
        return AIMessage(content="done")

    monkeypatch.setattr(w, "invoke_llm", fake_invoke)
    monkeypatch.setattr(w, "_agent_may_call_tools", lambda *_a, **_k: True)

    app = fresh_memory_graph()
    cfg = fresh_thread_config(suffix=f"intent-htc-{uuid.uuid4().hex[:8]}")
    payload = initial_turn_input("report on selftest")
    payload["diary_source"] = "intent"

    with intent_background_pass():
        out = app.invoke(payload, config=cfg)

    tool_msgs = [
        m for m in out["messages"] if getattr(m, "name", None) == "message_user"
    ]
    assert tool_msgs
    assert "Queued" in str(tool_msgs[0].content)
    assert _outbox_drafts() == [report]
    assert "outreach_queued" in _mind_kinds()
    _cleanup_agent_rows()


@pytest.mark.integration
@pytest.mark.db
def test_outreach_otchetitsya_unfulfilled_no_invented_outbox(
    disposable_postgres, monkeypatch
):
    """Paraphrase without "me" must still trip outreach gate (incident text)."""
    _cleanup_agent_rows()
    text = "Run a self-check (selftest) and report the results back."
    with get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO intents (agent_id, text, run_at, status)
                VALUES (%s, %s, now(), 'due')
                RETURNING id::text
                """,
                (settings.default_agent_id, text),
            )
            intent_id = cur.fetchone()[0]
        conn.commit()

    def _turn(message, *, thread_id, session_uuid, diary_source=None):
        from src.agent_runner import AgentTurnResult

        return AgentTurnResult(
            reply="<tool_call><function=message_user>…",
            tools_used=(),
            elapsed_ms=1.0,
        )

    monkeypatch.setattr("src.agent_runner.run_agent_turn_detailed", _turn)
    events: list[str] = []
    monkeypatch.setattr(
        intent_pass_mod,
        "log_event",
        lambda event, **_f: events.append(event),
    )
    metrics: list[str] = []
    monkeypatch.setattr(
        intent_pass_mod,
        "record_pulse_intent_run",
        lambda result: metrics.append(result),
    )

    assert intent_pass_mod.run_due_intent_if_any() is True
    assert _intent_status(intent_id) == "done"
    assert "intent.outreach_unfulfilled" in events
    assert metrics == ["unfulfilled"]
    assert _outbox_drafts() == []
    _cleanup_agent_rows()


@pytest.mark.integration
@pytest.mark.db
def test_schedule_mark_due_run_queues_and_sends(
    disposable_postgres, reset_outreach_state, monkeypatch
):
    """Full Pulse seam: schedule → heart mark_due → run → outbox → telegram mock."""
    _cleanup_agent_rows()
    marker = f"pulse-chain-{uuid.uuid4().hex[:10]}"
    intent_text = f"write me {marker}"

    scheduled = schedule_intent_impl(intent_text, "in 5 minutes")
    assert scheduled.startswith("Scheduled:"), scheduled

    with get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id::text, status FROM intents
                WHERE agent_id = %s AND text = %s
                """,
                (settings.default_agent_id, intent_text),
            )
            row = cur.fetchone()
        assert row is not None
        intent_id, status = row
        assert status == "pending"
        # Force due without waiting on the clock.
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE intents
                SET run_at = now() - interval '1 minute'
                WHERE id = %s::uuid
                """,
                (intent_id,),
            )
        mark_due_intents(conn)
        conn.commit()

    assert _intent_status(intent_id) == "due"
    assert "intent_scheduled" in _mind_kinds()

    def _turn(message, *, thread_id, session_uuid, diary_source=None):
        from src.agent_runner import AgentTurnResult
        from src.reverie.background import intent_background_pass as _intent_pass
        from src.reverie.background import reverie_background_pass
        from src.tools.message_user import message_user_impl

        with reverie_background_pass(), _intent_pass():
            message_user_impl(f"Report: {marker}")
        return AgentTurnResult(
            reply="stray", tools_used=("message_user",), elapsed_ms=2.0
        )

    monkeypatch.setattr("src.agent_runner.run_agent_turn_detailed", _turn)
    assert intent_pass_mod.run_due_intent_if_any() is True
    assert _intent_status(intent_id) == "done"
    assert _outbox_drafts() == [f"Report: {marker}"]
    kinds = _mind_kinds()
    assert "intent_run" in kinds
    assert "outreach_queued" in kinds
    assert "agent_out" not in kinds

    monkeypatch.setattr(settings, "telegram_owner_chat_id", 1, raising=False)
    monkeypatch.setattr(settings, "telegram_bot_token", "test-token", raising=False)

    with get_pool().connection() as conn:
        with patch.object(outreach_mod, "_send_telegram", return_value=True) as send:
            outreach_mod.send_outreach_if_due(conn)
        assert send.called
        with conn.cursor() as cur:
            cur.execute(
                "SELECT status FROM outbox WHERE agent_id = %s AND draft_text = %s",
                (settings.default_agent_id, f"Report: {marker}"),
            )
            assert cur.fetchone()[0] == "sent"
        conn.commit()

    assert "outreach_sent" in _mind_kinds()
    _cleanup_agent_rows()


@pytest.mark.integration
@pytest.mark.db
def test_mind_hub_skips_agent_out_on_intent_thread(disposable_postgres, monkeypatch):
    """False «reported» signal: intent visible reply must not become agent_out."""
    from src.mind.hub import record_agent_out, record_intent_run

    _cleanup_agent_rows()
    record_agent_out(
        reply="Intent completed on time…",
        thread_id=f"intent-{uuid.uuid4().hex[:8]}",
        channel="repl",
    )
    record_intent_run(text="report on selftest")
    kinds = _mind_kinds()
    assert "agent_out" not in kinds
    assert "intent_run" in kinds
    _cleanup_agent_rows()


@pytest.mark.integration
@pytest.mark.db
def test_outreach_fulfillment_after_corrective_queues_once(
    disposable_postgres, monkeypatch
):
    """Corrective round that actually calls message_user → one outbox, done."""
    _cleanup_agent_rows()
    text = "report the selftest results"
    with get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO intents (agent_id, text, run_at, status)
                VALUES (%s, %s, now(), 'due')
                RETURNING id::text
                """,
                (settings.default_agent_id, text),
            )
            intent_id = cur.fetchone()[0]
        conn.commit()

    marker = f"outreach-ok-{uuid.uuid4().hex[:8]}"
    calls: list[bool] = []

    def _turn(message, *, thread_id, session_uuid, diary_source=None):
        from src.agent_runner import AgentTurnResult
        from src.reverie.background import intent_background_pass as _intent_pass
        from src.reverie.background import reverie_background_pass
        from src.tools.message_user import message_user_impl

        corrective = "message_user" in (message or "") and "called successfully" in (
            message or ""
        )
        calls.append(corrective)
        if corrective:
            with reverie_background_pass(), _intent_pass():
                message_user_impl(marker)
        return AgentTurnResult(reply="", tools_used=(), elapsed_ms=1.0)

    monkeypatch.setattr("src.agent_runner.run_agent_turn_detailed", _turn)
    assert intent_pass_mod.run_due_intent_if_any() is True
    assert calls == [False, True]
    assert _intent_status(intent_id) == "done"
    assert _outbox_drafts() == [marker]
    _cleanup_agent_rows()
