"""Unit tests for solitary reflection cycle."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.agent_runner import AgentTurnResult
from src.config import settings
from src.reverie import solitary as solitary_mod
from src.reverie.cancel import ReverieCancelToken


@pytest.fixture(autouse=True)
def _no_outbox_db(monkeypatch):
    """Keep all outbox I/O off the real DB in unit tests.

    Critical: `.env` may have OUTREACH_ENABLED=1, so an unmocked fallback would
    call the real `message_user_impl` → INSERT into the production `agent_db`
    and Heart would deliver test payloads ("x"/"ok") to USER. Mock the queue by
    default; tests that assert queuing override `message_user_impl` themselves.
    """
    monkeypatch.setattr("src.tools.message_user.count_outbox", lambda _aid: 0)
    monkeypatch.setattr(
        "src.tools.message_user.stamp_pending_outbox_footer", lambda *_a, **_k: 0
    )
    monkeypatch.setattr(
        "src.tools.message_user.message_user_impl",
        lambda *_a, **_k: "Queued for delivery to USER.",
    )


@pytest.fixture
def solitary_outbox_allowed(monkeypatch):
    """Allow solitary outbox queue in unit tests (outreach on, no gates left)."""
    monkeypatch.setattr(settings, "outreach_enabled", True, raising=False)


@pytest.mark.unit
def test_run_solitary_skip_disabled(monkeypatch, reset_solitary_state):
    monkeypatch.setattr(settings, "solitary_enabled", False, raising=False)
    assert solitary_mod.run_solitary_pass_if_due() == "skip_disabled"


@pytest.mark.unit
def test_run_solitary_skip_interval(monkeypatch, reset_solitary_state):
    from src.pulse import rhythm as rhythm_mod
    from src.pulse.rhythm import mark_solitary_ran

    monkeypatch.setattr(settings, "solitary_enabled", True, raising=False)
    monkeypatch.setattr(settings, "solitary_interval_min", 10, raising=False)
    monkeypatch.setattr(
        rhythm_mod, "_last_solitary_epoch", lambda: rhythm_mod._last_solitary_at
    )
    mark_solitary_ran(1_000_000.0)
    monkeypatch.setattr("time.time", lambda: 1_000_100.0)
    assert solitary_mod.run_solitary_pass_if_due() == "skip_interval"


@pytest.mark.unit
def test_hygiene_lane_applies_hygiene_done(monkeypatch, reset_solitary_state):
    """A-2 close-loop: completed hygiene solitary pass must burn debt."""
    token = ReverieCancelToken()
    outcomes: list[str] = []
    monkeypatch.setattr(
        "src.policy_knobs.apply_outcome_safe",
        lambda sig, **_k: outcomes.append(sig),
    )
    monkeypatch.setattr("src.reverie.prompts.last_solitary_lane_id", lambda: "hygiene")
    monkeypatch.setattr(
        "src.reverie.attention.solitary_used_attention_cost",
        lambda **_k: False,
    )
    monkeypatch.setattr(
        "src.agent_runner.run_agent_turn_detailed",
        lambda *_a, **_k: AgentTurnResult(
            reply="hygiene done",
            tools_used=("forget_memory",),
            elapsed_ms=1.0,
        ),
    )
    assert solitary_mod._do_solitary_pass(token) == "ran"
    assert "hygiene_done" in outcomes


@pytest.mark.unit
def test_hygiene_lane_search_only_does_not_burn_debt(monkeypatch, reset_solitary_state):
    """Search alone is not close-loop hygiene — must not fire hygiene_done."""
    token = ReverieCancelToken()
    outcomes: list[str] = []
    monkeypatch.setattr(
        "src.policy_knobs.apply_outcome_safe",
        lambda sig, **_k: outcomes.append(sig),
    )
    monkeypatch.setattr("src.reverie.prompts.last_solitary_lane_id", lambda: "hygiene")
    monkeypatch.setattr(
        "src.reverie.attention.solitary_used_attention_cost",
        lambda **_k: False,
    )
    monkeypatch.setattr(
        "src.agent_runner.run_agent_turn_detailed",
        lambda *_a, **_k: AgentTurnResult(
            reply="checked memory",
            tools_used=("search_memory", "search_by_time"),
            elapsed_ms=1.0,
        ),
    )
    assert solitary_mod._do_solitary_pass(token) == "ran"
    assert "hygiene_done" not in outcomes


@pytest.mark.unit
def test_hygiene_lane_without_tools_does_not_burn_debt(
    monkeypatch, reset_solitary_state
):
    token = ReverieCancelToken()
    outcomes: list[str] = []
    monkeypatch.setattr(
        "src.policy_knobs.apply_outcome_safe",
        lambda sig, **_k: outcomes.append(sig),
    )
    monkeypatch.setattr("src.reverie.prompts.last_solitary_lane_id", lambda: "hygiene")
    monkeypatch.setattr(
        "src.reverie.attention.solitary_used_attention_cost",
        lambda **_k: False,
    )
    monkeypatch.setattr(
        "src.agent_runner.run_agent_turn_detailed",
        lambda *_a, **_k: AgentTurnResult(
            reply="thought",
            tools_used=("affirm_identity",),
            elapsed_ms=1.0,
        ),
    )
    assert solitary_mod._do_solitary_pass(token) == "ran"
    assert "hygiene_done" not in outcomes


@pytest.mark.unit
def test_run_solitary_skip_live_busy(monkeypatch, reset_solitary_state):
    from src.pulse import rhythm as rhythm_mod

    monkeypatch.setattr(settings, "solitary_enabled", True, raising=False)
    monkeypatch.setattr(settings, "solitary_interval_min", 0, raising=False)
    monkeypatch.setattr(rhythm_mod, "_last_solitary_epoch", lambda: 0.0)
    monkeypatch.setattr(
        "src.policy_knobs.solitary_deferred_for_live_busy", lambda: True
    )
    outcomes: list[str] = []
    monkeypatch.setattr(
        "src.policy_knobs.apply_outcome_safe",
        lambda sig, **_k: outcomes.append(sig),
    )
    assert solitary_mod.run_solitary_pass_if_due() == "skip_live_busy"
    assert "attention_deferred" in outcomes


@pytest.mark.unit
def test_do_solitary_plain_ran_does_not_auto_queue(
    monkeypatch, reset_solitary_state, solitary_outbox_allowed
):
    """Visible solitary reply without message_user stays internal — no TG spam."""
    token = ReverieCancelToken()
    monkeypatch.setattr(settings, "outreach_enabled", True, raising=False)
    queued: list[str] = []

    def _impl(text, **_k):
        queued.append(text)
        return "Queued for delivery to USER."

    monkeypatch.setattr("src.tools.message_user.message_user_impl", _impl)
    monkeypatch.setattr(
        "src.agent_runner.run_agent_turn_detailed",
        lambda *_a, **_k: AgentTurnResult(
            reply="I will not write now. Internal.",
            tools_used=("save_memory", "search_diary"),
            elapsed_ms=1.0,
            thread_id="solitary-abc",
        ),
    )
    assert solitary_mod._do_solitary_pass(token) == "ran"
    assert queued == []


@pytest.mark.unit
def test_do_solitary_does_not_deliver_empty_turn_fallback(
    monkeypatch, reset_solitary_state, solitary_outbox_allowed
):
    """An empty-turn fallback (tool dump) must never reach the outreach fallback."""
    token = ReverieCancelToken()
    monkeypatch.setattr(settings, "outreach_enabled", True, raising=False)
    queued: list[str] = []
    monkeypatch.setattr(
        "src.tools.message_user.message_user_impl",
        lambda text, **_k: (queued.append(text), "Queued")[1],
    )
    monkeypatch.setattr(
        "src.agent_runner.run_agent_turn_detailed",
        lambda *_a, **_k: AgentTurnResult(
            reply="No reply text. Tool results: ...",
            tools_used=(),
            elapsed_ms=1.0,
            reply_is_fallback=True,
        ),
    )
    assert solitary_mod._do_solitary_pass(token) == "ran"
    assert queued == []


@pytest.mark.unit
def test_do_solitary_flags_model_queued_when_outbox_grew(
    monkeypatch, reset_solitary_state, solitary_outbox_allowed
):
    """message_user queued during the pass → fallback is told model_queued=True."""
    token = ReverieCancelToken()
    monkeypatch.setattr(settings, "outreach_enabled", True, raising=False)
    counts = iter([0, 1])
    monkeypatch.setattr(
        "src.tools.message_user.count_outbox", lambda _aid: next(counts)
    )
    queued: list[str] = []
    monkeypatch.setattr(
        "src.tools.message_user.message_user_impl",
        lambda text, **_k: (queued.append(text), "Queued")[1],
    )
    monkeypatch.setattr(
        "src.agent_runner.run_agent_turn_detailed",
        lambda *_a, **_k: AgentTurnResult(
            reply="reflection", tools_used=("search_diary",), elapsed_ms=2140.0
        ),
    )
    stamped: list[str] = []
    monkeypatch.setattr(
        "src.tools.message_user.stamp_pending_outbox_footer",
        lambda _aid, footer: stamped.append(footer) or 1,
    )
    assert solitary_mod._do_solitary_pass(token) == "ran"
    assert queued == []  # model already queued → no fallback
    assert stamped and "→ turn" in stamped[0] and "search_diary" in stamped[0]


@pytest.mark.unit
def test_do_solitary_fail_closed_when_count_outbox_errors(
    monkeypatch, reset_solitary_state, solitary_outbox_allowed
):
    """DB count failure → model_queued=True so fallback does not duplicate."""
    token = ReverieCancelToken()
    monkeypatch.setattr(settings, "outreach_enabled", True, raising=False)
    monkeypatch.setattr("src.tools.message_user.count_outbox", lambda _aid: None)
    queued: list[str] = []
    monkeypatch.setattr(
        "src.tools.message_user.message_user_impl",
        lambda text, **_k: (queued.append(text), "Queued")[1],
    )
    monkeypatch.setattr(
        "src.agent_runner.run_agent_turn_detailed",
        lambda *_a, **_k: AgentTurnResult(
            reply="reflection", tools_used=(), elapsed_ms=1.0
        ),
    )
    assert solitary_mod._do_solitary_pass(token) == "ran"
    assert queued == []


@pytest.mark.unit
def test_do_solitary_cancelled_before_turn(reset_solitary_state):
    token = ReverieCancelToken()
    token.cancel()
    with patch("src.agent_runner.run_agent_turn_detailed") as turn:
        assert solitary_mod._do_solitary_pass(token) == "cancelled"
        turn.assert_not_called()


@pytest.mark.unit
def test_do_solitary_keeps_token_during_turn(
    monkeypatch, reset_solitary_state, solitary_outbox_allowed
):
    """Token stays registered for the whole pass — no early release before the turn."""
    token = ReverieCancelToken()
    order: list[str] = []

    monkeypatch.setattr(
        "src.reverie.cancel.end_reverie_cycle",
        lambda _tok: order.append("end_reverie"),
    )

    def _turn(*_a, **_k):
        order.append("turn")
        return AgentTurnResult(reply="ok", tools_used=(), elapsed_ms=1.0)

    monkeypatch.setattr("src.agent_runner.run_agent_turn_detailed", _turn)
    assert solitary_mod._do_solitary_pass(token) == "ran"
    assert order == ["turn"]


@pytest.mark.unit
def test_do_solitary_cancelled_mid_turn(monkeypatch, reset_solitary_state):
    """Live turn preempts the running pass via ReverieCancelledError."""
    from src.reverie.cancel import ReverieCancelledError

    token = ReverieCancelToken()

    def _turn(*_a, **_k):
        raise ReverieCancelledError("live turn preempted background reverie pass")

    monkeypatch.setattr("src.agent_runner.run_agent_turn_detailed", _turn)
    assert solitary_mod._do_solitary_pass(token) == "cancelled"


@pytest.mark.unit
def test_do_solitary_unique_thread_ids(
    monkeypatch, reset_solitary_state, solitary_outbox_allowed
):
    thread_ids: list[str] = []

    def _turn(_prompt, *, thread_id, session_uuid, diary_source=None):
        thread_ids.append(thread_id)
        return AgentTurnResult(reply="x", tools_used=(), elapsed_ms=1.0)

    monkeypatch.setattr("src.agent_runner.run_agent_turn_detailed", _turn)
    token = ReverieCancelToken()
    solitary_mod._do_solitary_pass(token)
    solitary_mod._do_solitary_pass(ReverieCancelToken())
    assert len(thread_ids) == 2
    assert all(t.startswith("solitary-") for t in thread_ids)
    assert thread_ids[0] != thread_ids[1]


@pytest.mark.unit
def test_do_solitary_empty_reply_still_ran(monkeypatch, reset_solitary_state):
    token = ReverieCancelToken()
    monkeypatch.setattr(
        "src.agent_runner.run_agent_turn_detailed",
        lambda *_a, **_k: AgentTurnResult(reply="  ", tools_used=(), elapsed_ms=1.0),
    )
    assert solitary_mod._do_solitary_pass(token) == "ran"


@pytest.mark.unit
def test_do_solitary_error_on_turn_exception(monkeypatch, reset_solitary_state):
    token = ReverieCancelToken()

    def _boom(*_a, **_k):
        raise RuntimeError("graph down")

    monkeypatch.setattr("src.agent_runner.run_agent_turn_detailed", _boom)
    assert solitary_mod._do_solitary_pass(token) == "error"


@pytest.mark.unit
def test_allow_solitary_fallback_always_false(monkeypatch, solitary_outbox_allowed):
    monkeypatch.setattr(settings, "outreach_enabled", True, raising=False)
    assert solitary_mod._allow_solitary_fallback("hi", True) is False
    assert (
        solitary_mod._allow_solitary_fallback("Hello, I thought about you.", False)
        is False
    )
