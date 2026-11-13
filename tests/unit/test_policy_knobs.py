"""Policy knobs — outcome nudges (no DB)."""

from __future__ import annotations

import pytest

from src.policy_knobs import PolicyKnobs


def _store_with_state(monkeypatch, state: dict[str, float]):
    store = PolicyKnobs(pool=object())
    monkeypatch.setattr(store, "get_all", lambda agent=None: dict(state))
    written: list[tuple[str, float]] = []

    def _set(key, value, *, agent=None, lesson_id=None):
        written.append((key, value))
        state[key] = value
        return True

    monkeypatch.setattr(store, "set", _set)
    return store, written


def test_apply_outcome_log_pile_lowers_trust(monkeypatch):
    state = {
        "unverified_rerank_factor": 0.35,
        "age_half_life_days": 180.0,
        "outreach_initiative": 0.5,
        "tool_curiosity": 0.5,
        "hygiene_debt": 0.0,
    }
    store, written = _store_with_state(monkeypatch, state)
    out = store.apply_outcome("log_pile")
    assert out["unverified_rerank_factor"] < 0.35
    assert out["age_half_life_days"] < 180.0
    assert out["hygiene_debt"] == pytest.approx(1.0)
    assert any(k == "unverified_rerank_factor" for k, _ in written)


def test_hygiene_done_recovers_trust_and_debt(monkeypatch):
    state = {
        "unverified_rerank_factor": 0.25,
        "age_half_life_days": 150.0,
        "outreach_initiative": 0.5,
        "tool_curiosity": 0.5,
        "hygiene_debt": 2.0,
    }
    store, _ = _store_with_state(monkeypatch, state)
    out = store.apply_outcome("hygiene_done")
    assert out["hygiene_debt"] == pytest.approx(1.0)
    assert out["unverified_rerank_factor"] > 0.25
    assert out["age_half_life_days"] > 150.0


def test_outreach_and_web_have_recovery_signals(monkeypatch):
    state = {
        "unverified_rerank_factor": 0.35,
        "age_half_life_days": 180.0,
        "outreach_initiative": 0.15,
        "tool_curiosity": 0.5,
        "hygiene_debt": 0.0,
    }
    store, _ = _store_with_state(monkeypatch, state)
    # `outreach_unfulfilled` no longer charges initiative. It fires when an
    # intent pass produced no message, and lowering the willingness to write
    # punished silence with more silence — while `message_user` refuses below
    # the gate floor and a delivered message is the only thing that raises the
    # knob. That made low initiative absorbing: observed live at 0.0 with two
    # `outreach.suppressed reason=low_initiative` attempts, and the agent had
    # not written first for two days.
    store.apply_outcome("outreach_unfulfilled")
    assert state["outreach_initiative"] == pytest.approx(0.15)
    store.apply_outcome("outreach_fulfilled")
    assert state["outreach_initiative"] == pytest.approx(0.2)
    store.apply_outcome("web_intent_unused")
    store.apply_outcome("web_intent_used")
    assert "web_verify_bias" not in state


def test_apply_outcome_skips_log_when_set_fails(monkeypatch):
    store = PolicyKnobs(pool=object())
    state = {
        "unverified_rerank_factor": 0.35,
        "age_half_life_days": 180.0,
        "outreach_initiative": 0.5,
        "tool_curiosity": 0.5,
        "hygiene_debt": 0.0,
    }
    monkeypatch.setattr(store, "get_all", lambda agent=None: dict(state))
    monkeypatch.setattr(store, "set", lambda *a, **k: False)
    events: list[str] = []
    monkeypatch.setattr(
        "src.policy_knobs.log_event",
        lambda event, **fields: events.append(event),
    )
    store.apply_outcome("log_pile")
    assert "policy.outcome" not in events


def test_sanitize_knob_value_clamps_poison():
    from src.policy_knobs import sanitize_knob_value

    assert sanitize_knob_value("age_half_life_days", 0.0) == 1.0
    assert sanitize_knob_value("age_half_life_days", float("nan")) == 180.0
    assert sanitize_knob_value("unverified_rerank_factor", -9.0) == 0.01
    assert sanitize_knob_value("hygiene_debt", 99.0) == 5.0


def test_solitary_deferred_fail_closed_on_idle_clock(monkeypatch):
    from src.policy_knobs import solitary_deferred_for_live_busy

    def _boom():
        raise RuntimeError("idle clock down")

    monkeypatch.setattr(
        "src.reverie.idle.minutes_since_last_user_turn",
        _boom,
    )
    assert solitary_deferred_for_live_busy() is True
