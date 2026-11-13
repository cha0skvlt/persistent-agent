"""Unit tests for BackgroundTurn orchestrator."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.reverie.background_deliver import (
    apply_outbox_delivery_tail,
    run_background_turn_and_deliver,
)
from src.reverie.cancel import ReverieCancelledError
from src.reverie.turn_kind import TurnKind


@pytest.fixture(autouse=True)
def _stub_outbox(monkeypatch):
    monkeypatch.setattr("src.tools.message_user.count_outbox", lambda _aid: 0)
    monkeypatch.setattr(
        "src.tools.message_user.stamp_pending_outbox_footer", lambda *_a, **_k: 0
    )
    # Default no-op so an unmocked fallback never writes to the real agent_db
    # (tests asserting queue override this).
    monkeypatch.setattr(
        "src.tools.message_user.message_user_impl",
        lambda *_a, **_k: "Queued for delivery to USER.",
    )


@pytest.mark.unit
def test_apply_tail_queues_fallback_when_not_model_queued(monkeypatch):
    queued: list[str] = []
    monkeypatch.setattr(
        "src.tools.message_user.message_user_impl",
        lambda text, **_k: (queued.append(text), "Queued for delivery to USER.")[1],
    )
    turn = SimpleNamespace(
        reply="Hello USER", reply_is_fallback=False, tools_used=(), elapsed_ms=1.0
    )
    result = apply_outbox_delivery_tail(
        kind=TurnKind.SOLITARY,
        turn=turn,
        outbox_before=0,
        allow_fallback=lambda r, mq: bool(r) and not mq,
        fallback_trigger="solitary_say",
    )
    assert result.model_queued is False
    assert result.fallback_queued is True
    assert queued == ["Hello USER"]
    assert result.result == "ran"


@pytest.mark.unit
def test_apply_tail_skips_fallback_when_outbox_grew(monkeypatch):
    counts = iter([0, 1])  # before unused here; after via count_outbox in tail
    monkeypatch.setattr(
        "src.tools.message_user.count_outbox", lambda _aid: next(counts, 1)
    )
    queued: list[str] = []
    monkeypatch.setattr(
        "src.tools.message_user.message_user_impl",
        lambda text, **_k: (queued.append(text), "Queued")[1],
    )
    turn = SimpleNamespace(
        reply="x", reply_is_fallback=False, tools_used=(), elapsed_ms=1
    )
    # outbox_before=0, after=1 → grew
    monkeypatch.setattr("src.tools.message_user.count_outbox", lambda _aid: 1)
    result = apply_outbox_delivery_tail(
        kind=TurnKind.INTENT,
        turn=turn,
        outbox_before=0,
        allow_fallback=lambda r, mq: bool(r) and not mq,
    )
    assert result.model_queued is True
    assert result.fallback_queued is False
    assert queued == []
    assert result.result == "done"


@pytest.mark.unit
def test_apply_tail_fail_closed_on_unknown_count(monkeypatch):
    monkeypatch.setattr("src.tools.message_user.count_outbox", lambda _aid: None)
    queued: list[str] = []
    monkeypatch.setattr(
        "src.tools.message_user.message_user_impl",
        lambda text, **_k: (queued.append(text), "Queued")[1],
    )
    turn = SimpleNamespace(
        reply="x", reply_is_fallback=False, tools_used=(), elapsed_ms=1
    )
    result = apply_outbox_delivery_tail(
        kind=TurnKind.SOLITARY,
        turn=turn,
        outbox_before=None,
        allow_fallback=lambda r, mq: bool(r) and not mq,
    )
    assert result.model_queued is True
    assert queued == []


@pytest.mark.unit
def test_apply_tail_drops_fallback_reply(monkeypatch):
    queued: list[str] = []
    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        "src.tools.message_user.message_user_impl",
        lambda text, **_k: (queued.append(text), "Queued")[1],
    )
    monkeypatch.setattr(
        "src.reverie.background_deliver.log_event",
        lambda event, **fields: events.append((event, fields)),
    )
    turn = SimpleNamespace(
        reply="Operator: No visible model output",
        reply_is_fallback=True,
        tools_used=(),
        elapsed_ms=1,
    )
    result = apply_outbox_delivery_tail(
        kind=TurnKind.SOLITARY,
        turn=turn,
        outbox_before=0,
        allow_fallback=lambda r, mq: bool(r) and not mq,
    )
    assert result.reply == ""
    assert queued == []
    assert any(e[0] == "background.silence" for e in events)


@pytest.mark.unit
def test_apply_tail_drops_ollama_template_error(monkeypatch):
    """Ollama Jinja dump must never become proactive mail (even if flag missing)."""
    queued: list[str] = []
    monkeypatch.setattr(
        "src.tools.message_user.message_user_impl",
        lambda text, **_k: (queued.append(text), "Queued")[1],
    )
    err = (
        '{"error":{"code":400,"message":"Unable to generate parser for this template. '
        'Automatic parser generation failed: No user query found in messages.",'
        '"type":"invalid_request_error"}}'
    )
    turn = SimpleNamespace(
        reply=err, reply_is_fallback=False, tools_used=(), elapsed_ms=553000.0
    )
    result = apply_outbox_delivery_tail(
        kind=TurnKind.SOLITARY,
        turn=turn,
        outbox_before=0,
        allow_fallback=lambda r, mq: bool(r) and not mq,
    )
    assert result.reply == ""
    assert queued == []
    assert result.fallback_queued is False


@pytest.mark.unit
def test_run_background_cancelled(monkeypatch):
    stamped: list[str] = []
    monkeypatch.setattr(
        "src.tools.message_user.stamp_pending_outbox_footer",
        lambda _aid, footer: stamped.append(footer) or 1,
    )
    # No outbox growth → no orphan stamp.
    monkeypatch.setattr("src.tools.message_user.count_outbox", lambda _aid: 0)

    def _run():
        raise ReverieCancelledError("preempt")

    out = run_background_turn_and_deliver(
        kind=TurnKind.SOLITARY, run_turn=_run, on_cancelled="cancelled"
    )
    assert out.result == "cancelled"
    assert out.turn is None
    assert stamped == []


@pytest.mark.unit
def test_run_background_cancelled_stamps_message_user_footer(monkeypatch):
    """Preempt after message_user — stamp orphan with wall-clock duration + tool."""
    stamped: list[str] = []
    counts = iter([0, 1])  # before, after cancel
    monkeypatch.setattr(
        "src.tools.message_user.count_outbox", lambda _aid: next(counts)
    )
    monkeypatch.setattr(
        "src.tools.message_user.stamp_pending_outbox_footer",
        lambda _aid, footer: stamped.append(footer) or 1,
    )
    ticks = iter([100.0, 103.5])  # 3.5s wall clock
    monkeypatch.setattr("time.perf_counter", lambda: next(ticks))

    def _run():
        raise ReverieCancelledError("preempt")

    out = run_background_turn_and_deliver(
        kind=TurnKind.SOLITARY, run_turn=_run, on_cancelled="cancelled"
    )
    assert out.result == "cancelled"
    assert stamped == ["**→ turn 3.5s**  message_user"]


@pytest.mark.unit
def test_run_background_error(monkeypatch):
    def _run():
        raise RuntimeError("boom")

    out = run_background_turn_and_deliver(
        kind=TurnKind.INTENT, run_turn=_run, on_error="failed"
    )
    assert out.result == "failed"


@pytest.mark.unit
def test_run_background_stamps_footer(monkeypatch):
    stamped: list[str] = []
    monkeypatch.setattr(
        "src.tools.message_user.stamp_pending_outbox_footer",
        lambda _aid, footer: stamped.append(footer) or 1,
    )
    monkeypatch.setattr(
        "src.tools.message_user.message_user_impl",
        lambda *_a, **_k: "Not sent: cooldown",
    )

    def _run():
        return SimpleNamespace(
            reply="hi",
            reply_is_fallback=False,
            tools_used=("search_diary",),
            elapsed_ms=2100.0,
        )

    out = run_background_turn_and_deliver(
        kind=TurnKind.SOLITARY,
        run_turn=_run,
        allow_fallback=lambda *_: True,
    )
    assert out.result == "ran"
    assert stamped and "→ turn" in stamped[0] and "search_diary" in stamped[0]
