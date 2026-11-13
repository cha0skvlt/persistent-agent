"""Attention economy++ — knobs bind to real solitary costs."""

from __future__ import annotations

import pytest

from src.graph import tool_round_limit as trl
from src.policy_knobs import (
    PolicyKnobs,
    essay_extra_tool_rounds,
    live_busy_defer_minutes,
    solitary_deferred_for_live_busy,
)


def test_live_busy_defer_minutes_scales_with_reserve(monkeypatch):
    monkeypatch.setattr(
        "src.policy_knobs.effective_knobs",
        lambda **_: {"attention_reserve": 0.3},
    )
    assert live_busy_defer_minutes() == pytest.approx(1.5)
    monkeypatch.setattr(
        "src.policy_knobs.effective_knobs",
        lambda **_: {"attention_reserve": 0.0},
    )
    assert live_busy_defer_minutes() == 0.5
    monkeypatch.setattr(
        "src.policy_knobs.effective_knobs",
        lambda **_: {"attention_reserve": 1.0},
    )
    # Hard ceiling — no death-spiral defer window.
    assert live_busy_defer_minutes() == 3.0


def test_solitary_deferred_when_user_recent(monkeypatch):
    monkeypatch.setattr(
        "src.policy_knobs.effective_knobs",
        lambda **_: {"attention_reserve": 0.3},
    )
    monkeypatch.setattr(
        "src.reverie.idle.minutes_since_last_user_turn",
        lambda: 1.0,
    )
    assert solitary_deferred_for_live_busy() is True
    monkeypatch.setattr(
        "src.reverie.idle.minutes_since_last_user_turn",
        lambda: 10.0,
    )
    assert solitary_deferred_for_live_busy() is False


def test_essay_extra_rounds_solitary_only(monkeypatch):
    monkeypatch.setattr(
        "src.policy_knobs.effective_knobs",
        lambda **_: {"tool_curiosity": 0.9},
    )
    monkeypatch.setattr(
        "src.reverie.turn_kind.is_solitary_turn",
        lambda: False,
    )
    assert essay_extra_tool_rounds() == 0
    monkeypatch.setattr(
        "src.reverie.turn_kind.is_solitary_turn",
        lambda: True,
    )
    assert essay_extra_tool_rounds() == 2


def test_tool_cap_includes_essay(monkeypatch):
    monkeypatch.setattr(
        trl.config.settings, "max_tool_rounds_per_turn", 2, raising=False
    )
    monkeypatch.setattr(
        trl.config.settings, "reverie_max_tool_rounds", 2, raising=False
    )
    monkeypatch.setattr(
        "src.reverie.background.is_reverie_background_pass", lambda: True
    )
    monkeypatch.setattr("src.policy_knobs.curiosity_extra_tool_rounds", lambda: 0)
    monkeypatch.setattr("src.policy_knobs.essay_extra_tool_rounds", lambda: 2)
    assert trl.tool_round_cap_reached({"tool_rounds": 3}) is False
    assert trl.tool_round_cap_reached({"tool_rounds": 4}) is True


def test_tool_cap_essay_does_not_stack_curiosity(monkeypatch):
    monkeypatch.setattr(
        trl.config.settings, "max_tool_rounds_per_turn", 2, raising=False
    )
    monkeypatch.setattr(
        trl.config.settings, "reverie_max_tool_rounds", 2, raising=False
    )
    monkeypatch.setattr(
        "src.reverie.background.is_reverie_background_pass", lambda: True
    )
    monkeypatch.setattr("src.policy_knobs.curiosity_extra_tool_rounds", lambda: 1)
    monkeypatch.setattr("src.policy_knobs.essay_extra_tool_rounds", lambda: 2)
    # Cap = 2 + 2 (essay), not 2 + 1 + 2.
    assert trl.tool_round_cap_reached({"tool_rounds": 3}) is False
    assert trl.tool_round_cap_reached({"tool_rounds": 4}) is True
    assert trl.tool_round_cap_reached({"tool_rounds": 5}) is True


def test_solitary_wall_budget_scales_with_reserve(monkeypatch):
    from src.reverie.attention import solitary_wall_budget_sec

    monkeypatch.setattr(
        "src.policy_knobs.effective_knobs",
        lambda **_: {"attention_reserve": 0.0},
    )
    assert solitary_wall_budget_sec() == 120.0
    monkeypatch.setattr(
        "src.policy_knobs.effective_knobs",
        lambda **_: {"attention_reserve": 1.0},
    )
    assert solitary_wall_budget_sec() == 30.0


def test_attention_cost_uses_multi_tool_not_unique_cap(monkeypatch):
    from src.reverie.attention import solitary_used_attention_cost

    monkeypatch.setattr(
        "src.policy_knobs.effective_knobs",
        lambda **_: {"attention_reserve": 0.3},
    )
    assert (
        solitary_used_attention_cost(tools_used=("search_web",), elapsed_ms=100)
        is False
    )
    assert (
        solitary_used_attention_cost(
            tools_used=("search_web", "save_memory"), elapsed_ms=100
        )
        is True
    )


def test_curiosity_spent_outcome(monkeypatch):
    state = {
        "unverified_rerank_factor": 0.35,
        "age_half_life_days": 180.0,
        "outreach_initiative": 0.5,
        "tool_curiosity": 0.7,
        "hygiene_debt": 0.0,
        "attention_reserve": 0.3,
    }
    store = PolicyKnobs(pool=object())
    monkeypatch.setattr(store, "get_all", lambda agent=None: dict(state))

    def _set(key, value, *, agent=None, lesson_id=None):
        state[key] = value
        return True

    monkeypatch.setattr(store, "set", _set)
    # `curiosity_spent` no longer charges for using an allotted tool round: the
    # -0.03 penalty outweighed the only refill (+0.02 on `lesson_saved`), so the
    # knob ratcheted to 0.0 — pinning the motive to help_complete and switching
    # off both extra-round grants. Spending is now log-only.
    store.apply_outcome("curiosity_spent")
    assert state["tool_curiosity"] == pytest.approx(0.7)
    store.apply_outcome("attention_deferred")
    # Recover toward allowing solitary — never widen the defer window.
    assert state["attention_reserve"] == pytest.approx(0.29)
