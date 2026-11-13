"""Unit tests for Pulse intent pass — mocked DB, no disposable_postgres."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from src.config import settings
from src.reverie import intent_pass as ip


@pytest.fixture(autouse=True)
def _intent_pass_settings(monkeypatch):
    monkeypatch.setattr(settings, "agent_tz", "UTC", raising=False)
    monkeypatch.setattr(settings, "default_agent_id", "agent", raising=False)
    monkeypatch.setattr(settings, "pulse_enabled", True, raising=False)
    # Patch the names ``intent_pass`` actually calls (module-level imports).
    monkeypatch.setattr(ip, "count_outbox", lambda _aid: 0)
    monkeypatch.setattr(ip, "apply_outbox_delivery_tail", lambda **_k: None)
    monkeypatch.setattr("src.tools.message_user.count_outbox", lambda _aid: 0)
    monkeypatch.setattr(
        "src.tools.message_user.stamp_pending_outbox_footer", lambda *_a, **_k: 0
    )
    # No-op queue so an unmocked fallback never hits the real agent_db.
    monkeypatch.setattr(
        "src.tools.message_user.message_user_impl",
        lambda *_a, **_k: "Queued for delivery to USER.",
    )


@pytest.mark.unit
def test_intent_prompt_allows_selftest():
    run_at = datetime(2026, 7, 8, 9, 0, tzinfo=timezone.utc)
    prompt = ip._build_intent_prompt(text="selftest and write to me", run_at=run_at)
    assert "run_selftest is available" in prompt
    assert "live only" not in prompt


@pytest.mark.unit
def test_run_due_intent_cancelled_backoff_pending(monkeypatch):
    run_at = datetime(2026, 7, 8, 9, 0, tzinfo=timezone.utc)
    claimed = ("id-c", "check mail", run_at, None)
    status_updates: list[tuple] = []
    events: list[tuple[str, dict]] = []

    monkeypatch.setattr(ip, "_claim_due_intent", lambda: claimed)
    monkeypatch.setattr(
        ip,
        "_set_intent_status",
        lambda intent_id, *, status, run_at=None, reset_attempts=False: status_updates.append(
            (intent_id, status, run_at)
        ),
    )
    monkeypatch.setattr(
        ip, "_run_intent_graph", lambda *a, **k: ("cancelled", "", None, "t1")
    )
    monkeypatch.setattr(
        "src.reverie.intent_pass.log_event",
        lambda event, **fields: events.append((event, fields)),
    )
    monkeypatch.setattr(
        "src.reverie.intent_pass.record_pulse_intent_run",
        lambda result: None,
    )

    assert ip.run_due_intent_if_any() is True
    assert status_updates[0][0] == "id-c"
    assert status_updates[0][1] == "pending"
    assert status_updates[0][2] is not None
    assert any(e[0] == "intent.cancel_backoff" for e in events)
    assert any(
        e[0] == "intent.run" and e[1].get("result") == "cancelled" for e in events
    )


@pytest.mark.unit
def test_outreach_corrective_cancel_uses_backoff(monkeypatch):
    """Corrective cancel must park pending+2m, not snap to due (P1 regression)."""
    run_at = datetime(2026, 7, 8, 9, 0, tzinfo=timezone.utc)
    status_updates: list[tuple] = []
    events: list[tuple[str, dict]] = []
    calls: list[dict] = []

    monkeypatch.setattr(
        ip,
        "_claim_due_intent",
        lambda: ("id-corr-c", "write the selftest results", run_at, None),
    )
    monkeypatch.setattr(
        ip,
        "_set_intent_status",
        lambda intent_id, *, status, run_at=None, reset_attempts=False: status_updates.append(
            (intent_id, status, run_at)
        ),
    )

    def _graph(token, *, intent_id, text, run_at, thread_id=None, corrective=False):
        calls.append({"corrective": corrective})
        if corrective:
            return "cancelled", "", None, thread_id or "t1"
        return "done", "forgot tool", None, "t1"

    monkeypatch.setattr(ip, "_run_intent_graph", _graph)
    monkeypatch.setattr(ip, "count_outbox", lambda _aid: 0)
    monkeypatch.setattr(
        "src.reverie.intent_pass.log_event",
        lambda event, **fields: events.append((event, fields)),
    )
    monkeypatch.setattr(
        "src.reverie.intent_pass.record_pulse_intent_run",
        lambda result: None,
    )

    assert ip.run_due_intent_if_any() is True
    assert len(calls) == 2
    assert calls[1]["corrective"] is True
    assert status_updates[0][1] == "pending"
    assert status_updates[0][2] is not None
    assert any(e[0] == "intent.cancel_backoff" for e in events)
    assert not any(s == "due" for _, s, _ in status_updates)


@pytest.mark.unit
def test_run_due_intent_claims_and_completes(monkeypatch):
    run_at = datetime(2026, 7, 8, 9, 0, tzinfo=timezone.utc)
    claimed = ("id-1", "check mail", run_at, None)
    status_updates: list[tuple] = []
    events: list[tuple[str, dict]] = []
    metric_results: list[str] = []

    monkeypatch.setattr(ip, "_claim_due_intent", lambda: claimed)
    monkeypatch.setattr(
        ip,
        "_set_intent_status",
        lambda intent_id, *, status, run_at=None, reset_attempts=False: status_updates.append(
            (intent_id, status, run_at, reset_attempts)
        ),
    )
    monkeypatch.setattr(
        ip, "_run_intent_graph", lambda *a, **k: ("done", "", None, "t1")
    )
    monkeypatch.setattr(
        "src.mind.hub.record_intent_run",
        lambda **_: None,
    )
    monkeypatch.setattr(
        "src.reverie.intent_pass.log_event",
        lambda event, **fields: events.append((event, fields)),
    )
    monkeypatch.setattr(
        "src.reverie.intent_pass.record_pulse_intent_run",
        lambda result: metric_results.append(result),
    )

    assert ip.run_due_intent_if_any() is True
    assert status_updates == [("id-1", "done", None, True)]
    assert events == [("intent.run", {"result": "done", "intent_id": "id-1"})]
    assert metric_results == ["done"]


@pytest.mark.unit
def test_run_due_intent_recur_reschedules(monkeypatch):
    run_at = datetime(2026, 7, 8, 9, 0, tzinfo=timezone.utc)
    fixed_now = datetime(2026, 7, 8, 21, 30, tzinfo=ZoneInfo("UTC"))
    status_updates: list[tuple] = []

    monkeypatch.setattr(
        ip,
        "_claim_due_intent",
        lambda: ("id-2", "evening", run_at, "daily 21:00"),
    )
    monkeypatch.setattr(
        ip,
        "_set_intent_status",
        lambda intent_id, *, status, run_at=None, reset_attempts=False: status_updates.append(
            (intent_id, status, run_at, reset_attempts)
        ),
    )
    monkeypatch.setattr(
        ip, "_run_intent_graph", lambda *a, **k: ("done", "", None, "t2")
    )
    monkeypatch.setattr("src.mind.hub.record_intent_run", lambda **_: None)
    monkeypatch.setattr("src.reverie.intent_pass.log_event", lambda *a, **k: None)
    monkeypatch.setattr(
        "src.reverie.intent_pass.record_pulse_intent_run", lambda *_: None
    )

    with patch.object(ip, "datetime") as mock_dt:
        mock_dt.now.return_value = fixed_now
        mock_dt.side_effect = lambda *a, **k: datetime(*a, **k)
        assert ip.run_due_intent_if_any() is True

    assert status_updates[0][0] == "id-2"
    assert status_updates[0][1] == "pending"
    assert status_updates[0][2] > fixed_now
    assert status_updates[0][3] is True


@pytest.mark.unit
def test_run_due_intent_failed_parks(monkeypatch):
    monkeypatch.setattr(
        ip,
        "_claim_due_intent",
        lambda: ("id-3", "x", datetime.now(timezone.utc), None),
    )
    parked: list[str] = []
    monkeypatch.setattr(
        ip, "_park_failed_intent", lambda intent_id: parked.append(intent_id)
    )
    monkeypatch.setattr(
        ip, "_run_intent_graph", lambda *a, **k: ("failed", "", None, "t3")
    )
    monkeypatch.setattr("src.reverie.intent_pass.log_event", lambda *a, **k: None)
    monkeypatch.setattr(
        "src.reverie.intent_pass.record_pulse_intent_run", lambda *_: None
    )

    assert ip.run_due_intent_if_any() is True
    assert parked == ["id-3"]


@pytest.mark.unit
def test_run_intent_graph_drops_empty_turn_fallback(monkeypatch):
    """A fallback reply (tool dump) is not real prose → returned as empty."""
    from src.agent_runner import AgentTurnResult
    from src.reverie.cancel import ReverieCancelToken

    turn = AgentTurnResult(
        reply="No reply text. Tool results: ...",
        tools_used=(),
        elapsed_ms=1.0,
        reply_is_fallback=True,
    )
    monkeypatch.setattr(
        "src.agent_runner.run_agent_turn_detailed",
        lambda *a, **k: turn,
    )
    result, reply, got_turn, tid = ip._run_intent_graph(
        ReverieCancelToken(),
        intent_id="x",
        text="t",
        run_at=datetime.now(timezone.utc),
    )
    assert (result, reply, got_turn) == ("done", "", turn)
    assert tid.startswith("intent-")


@pytest.mark.unit
def test_run_intent_graph_cancelled_before_invoke():
    from src.reverie.cancel import ReverieCancelToken

    token = ReverieCancelToken()
    token.cancel()
    result, reply, turn, tid = ip._run_intent_graph(
        token,
        intent_id="x",
        text="t",
        run_at=datetime.now(timezone.utc),
    )
    assert (result, reply, turn) == ("cancelled", "", None)
    assert tid.startswith("intent-")


@pytest.mark.unit
def test_run_intent_graph_cancelled_from_graph(monkeypatch):
    from src.reverie.cancel import ReverieCancelledError, ReverieCancelToken

    def _boom(*a, **k):
        raise ReverieCancelledError()

    monkeypatch.setattr("src.agent_runner.run_agent_turn_detailed", _boom)
    token = ReverieCancelToken()
    result, reply, turn, _tid = ip._run_intent_graph(
        token,
        intent_id="x",
        text="t",
        run_at=datetime.now(timezone.utc),
    )
    assert (result, reply, turn) == ("cancelled", "", None)


@pytest.mark.unit
def test_run_intent_graph_failed_logs(monkeypatch):
    events: list[tuple[str, dict]] = []

    def _boom(*a, **k):
        raise RuntimeError("graph boom")

    monkeypatch.setattr("src.agent_runner.run_agent_turn_detailed", _boom)
    monkeypatch.setattr(
        "src.reverie.intent_pass.log_event",
        lambda event, **fields: events.append((event, fields)),
    )
    from src.reverie.cancel import ReverieCancelToken

    result, reply, turn, _tid = ip._run_intent_graph(
        ReverieCancelToken(),
        intent_id="bad-id",
        text="t",
        run_at=datetime.now(timezone.utc),
    )
    assert (result, reply, turn) == ("failed", "", None)
    assert events[0][0] == "intent.run_failed"
    assert events[0][1]["intent_id"] == "bad-id"


@pytest.mark.unit
def test_claim_due_intent_none_when_empty(monkeypatch):
    def _fake_pool():
        conn = MagicMock()
        cur = MagicMock()
        cur.fetchone.return_value = None
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cur)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        pool = MagicMock()
        pool.connection.return_value.__enter__ = MagicMock(return_value=conn)
        pool.connection.return_value.__exit__ = MagicMock(return_value=False)
        return pool

    with patch("src.db.pool.get_pool", _fake_pool):
        assert ip._claim_due_intent() is None


@pytest.mark.unit
def test_record_intent_run_hub(monkeypatch):
    recorded: dict = {}

    monkeypatch.setattr(settings, "mind_hub_enabled", True, raising=False)
    monkeypatch.setattr(
        "src.mind.hub._record",
        lambda **kw: recorded.update(kw),
    )
    from src.mind.hub import record_intent_run

    record_intent_run(text="check X")
    assert recorded["kind"] == "intent_run"
    assert recorded["channel"] == "pulse"
    assert recorded["content"] == "check X"


@pytest.mark.unit
def test_late_delta_hours_only():
    now = datetime(2026, 7, 8, 15, 0, tzinfo=ZoneInfo("UTC"))
    run_at = now - timedelta(hours=2)
    line = ip._late_delta_line(now=now, run_at=run_at)
    assert line == "The scheduled time passed 2h ago — take that into account."


@pytest.mark.unit
def test_is_outreach_shaped_verb_alone_counts():
    # Stored schedule text often drops "me" / USER.
    assert ip.is_outreach_shaped("in 2 min — write the selftest results") is True
    assert ip.is_outreach_shaped("write the PING_P3_NOMNE results") is True
    assert ip.is_outreach_shaped("in 2 min — write me the selftest results") is True
    assert ip.is_outreach_shaped("tell USER good morning") is True
    assert ip.is_outreach_shaped("notify USER of the status") is True
    assert ip.is_outreach_shaped("remind me about selftest") is True
    assert ip.is_outreach_shaped("run selftest and save it to memory") is False
    # Report verbs alone (paraphrase often drops "me" / USER).
    assert (
        ip.is_outreach_shaped(
            "Run a self-check (selftest) and report the results back."
        )
        is True
    )
    assert ip.is_outreach_shaped("report on the status") is True
    # Soft diary revisit — no outreach verb.
    assert (
        ip.is_outreach_shaped(
            "Return to the monument topic — check how the work reflects"
        )
        is False
    )
    # Substring traps from verb-alone gate: not outreach.
    assert ip.is_outreach_shaped("think about the dream topic") is False
    assert ip.is_outreach_shaped("I already wrote to the diary") is False
    assert ip.is_outreach_shaped("read what was written yesterday") is False
    # Infinitive still counts.
    assert ip.is_outreach_shaped("write USER the selftest results") is True


@pytest.mark.unit
def test_outreach_treats_unknown_outbox_count_as_not_queued(monkeypatch):
    """count_outbox → None must not skip outreach fulfillment (opposite of fallback fail-closed)."""
    run_at = datetime(2026, 7, 8, 9, 0, tzinfo=timezone.utc)
    calls: list[dict] = []
    events: list[tuple[str, dict]] = []

    monkeypatch.setattr(
        ip,
        "_claim_due_intent",
        lambda: ("id-none", "report the selftest results", run_at, None),
    )
    monkeypatch.setattr(ip, "_set_intent_status", lambda *_a, **_k: None)

    def _graph(token, *, intent_id, text, run_at, thread_id=None, corrective=False):
        calls.append({"corrective": corrective})
        return "done", "tool dump only", None, thread_id or "intent-none"

    monkeypatch.setattr(ip, "_run_intent_graph", _graph)
    monkeypatch.setattr(ip, "count_outbox", lambda _aid: None)
    monkeypatch.setattr("src.mind.hub.record_intent_run", lambda **_: None)
    monkeypatch.setattr(
        ip, "log_event", lambda event, **fields: events.append((event, fields))
    )
    monkeypatch.setattr(ip, "record_pulse_intent_run", lambda *_: None)

    assert ip.run_due_intent_if_any() is True
    assert len(calls) == 2
    assert calls[1]["corrective"] is True
    assert ("intent.outreach_unfulfilled", {"intent_id": "id-none"}) in events


@pytest.mark.unit
def test_outreach_retry_then_unfulfilled(monkeypatch):
    """Outreach-shaped, no message_user → one corrective round → unfulfilled."""
    run_at = datetime(2026, 7, 8, 9, 0, tzinfo=timezone.utc)
    calls: list[dict] = []
    events: list[tuple[str, dict]] = []
    metrics: list[str] = []
    statuses: list[str] = []

    monkeypatch.setattr(
        ip,
        "_claim_due_intent",
        lambda: ("id-outreach", "write me the selftest results", run_at, None),
    )
    monkeypatch.setattr(
        ip,
        "_set_intent_status",
        lambda _id, *, status, run_at=None, reset_attempts=False: statuses.append(
            status
        ),
    )

    def _graph(token, *, intent_id, text, run_at, thread_id=None, corrective=False):
        calls.append({"corrective": corrective, "thread_id": thread_id})
        tid = thread_id or "intent-abc"
        return "done", "I forgot the tool", None, tid

    monkeypatch.setattr(ip, "_run_intent_graph", _graph)
    monkeypatch.setattr(ip, "count_outbox", lambda _aid: 0)
    monkeypatch.setattr("src.mind.hub.record_intent_run", lambda **_: None)
    monkeypatch.setattr(
        ip, "log_event", lambda event, **fields: events.append((event, fields))
    )
    monkeypatch.setattr(ip, "record_pulse_intent_run", lambda r: metrics.append(r))

    assert ip.run_due_intent_if_any() is True
    assert len(calls) == 2
    assert calls[0]["corrective"] is False
    assert calls[1]["corrective"] is True
    assert calls[1]["thread_id"] == "intent-abc"
    assert statuses == ["done"]
    assert ("intent.outreach_unfulfilled", {"intent_id": "id-outreach"}) in events
    assert metrics == ["unfulfilled"]


@pytest.mark.unit
def test_outreach_gate_silent_when_not_outreach_shaped(monkeypatch):
    """No write+me → no corrective round; normal done path."""
    run_at = datetime(2026, 7, 8, 9, 0, tzinfo=timezone.utc)
    calls: list[dict] = []

    monkeypatch.setattr(
        ip,
        "_claim_due_intent",
        lambda: ("id-quiet", "run selftest", run_at, None),
    )
    monkeypatch.setattr(ip, "_set_intent_status", lambda *_a, **_k: None)

    def _graph(token, *, intent_id, text, run_at, thread_id=None, corrective=False):
        calls.append({"corrective": corrective})
        return "done", "", None, thread_id or "intent-q"

    monkeypatch.setattr(ip, "_run_intent_graph", _graph)
    monkeypatch.setattr(ip, "count_outbox", lambda _aid: 0)
    monkeypatch.setattr("src.mind.hub.record_intent_run", lambda **_: None)
    monkeypatch.setattr(ip, "log_event", lambda *a, **k: None)
    monkeypatch.setattr(ip, "record_pulse_intent_run", lambda *_: None)
    monkeypatch.setattr(
        "src.reverie.background_deliver.apply_outbox_delivery_tail",
        lambda **_: None,
    )

    assert ip.run_due_intent_if_any() is True
    assert len(calls) == 1
    assert calls[0]["corrective"] is False


@pytest.mark.unit
def test_outreach_no_retry_when_message_user_queued(monkeypatch):
    run_at = datetime(2026, 7, 8, 9, 0, tzinfo=timezone.utc)
    calls: list[dict] = []
    counts = iter([0, 1])  # before / after first pass

    monkeypatch.setattr(
        ip,
        "_claim_due_intent",
        lambda: ("id-ok", "tell USER ok", run_at, None),
    )
    monkeypatch.setattr(ip, "_set_intent_status", lambda *_a, **_k: None)

    def _graph(token, *, intent_id, text, run_at, thread_id=None, corrective=False):
        calls.append({"corrective": corrective})
        return "done", "", None, thread_id or "intent-ok"

    monkeypatch.setattr(ip, "_run_intent_graph", _graph)
    monkeypatch.setattr(ip, "count_outbox", lambda _aid: next(counts, 1))
    monkeypatch.setattr("src.mind.hub.record_intent_run", lambda **_: None)
    monkeypatch.setattr(ip, "log_event", lambda *a, **k: None)
    monkeypatch.setattr(ip, "record_pulse_intent_run", lambda *_: None)
    monkeypatch.setattr(
        "src.reverie.background_deliver.apply_outbox_delivery_tail",
        lambda **_: None,
    )

    assert ip.run_due_intent_if_any() is True
    assert len(calls) == 1
    assert calls[0]["corrective"] is False


@pytest.mark.unit
def test_model_queued_trusts_message_user_tool_when_count_unknown(monkeypatch):
    from types import SimpleNamespace

    from src.reverie.intent_pass import _model_queued_outbox

    monkeypatch.setattr(
        "src.reverie.intent_pass.count_outbox", lambda _aid: None, raising=True
    )
    turn = SimpleNamespace(tools_used=("search_web", "message_user"))
    assert _model_queued_outbox(0, turn) is True
    turn2 = SimpleNamespace(tools_used=("search_web",))
    assert _model_queued_outbox(0, turn2) is False


def test_outreach_corrective_succeeds_without_unfulfilled(monkeypatch):
    run_at = datetime(2026, 7, 8, 9, 0, tzinfo=timezone.utc)
    calls: list[dict] = []
    box = {"n": 0}

    monkeypatch.setattr(
        ip,
        "_claim_due_intent",
        lambda: ("id-fix", "write me the summary", run_at, None),
    )
    monkeypatch.setattr(ip, "_set_intent_status", lambda *_a, **_k: None)

    def _graph(token, *, intent_id, text, run_at, thread_id=None, corrective=False):
        calls.append({"corrective": corrective})
        if corrective:
            box["n"] = 1
        return "done", "", None, thread_id or "intent-fix"

    monkeypatch.setattr(ip, "_run_intent_graph", _graph)
    monkeypatch.setattr(ip, "count_outbox", lambda _aid: box["n"])
    monkeypatch.setattr("src.mind.hub.record_intent_run", lambda **_: None)
    events: list[str] = []
    monkeypatch.setattr(ip, "log_event", lambda e, **k: events.append(e))
    monkeypatch.setattr(ip, "record_pulse_intent_run", lambda *_: None)
    monkeypatch.setattr(
        "src.reverie.background_deliver.apply_outbox_delivery_tail",
        lambda **_: None,
    )

    assert ip.run_due_intent_if_any() is True
    assert len(calls) == 2
    assert "intent.outreach_unfulfilled" not in events
    assert "intent.run" in events
