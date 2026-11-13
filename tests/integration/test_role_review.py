"""Tests for offline role review and DPO gold pairs."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from src.dpo import role_review as rr
from src.graph import role_review_runner as rrr


@pytest.mark.unit
def test_parse_review_json_ok():
    raw = json.dumps(
        {
            "role_failed": True,
            "critique": "Corporate tone.",
            "gold_reply": "Fixed AGENT prose.",
        }
    )
    out = rr._parse_review_json(raw)
    assert out is not None
    assert out.role_failed is True
    assert out.gold_reply == "Fixed AGENT prose."


@pytest.mark.unit
def test_parse_review_json_role_ok_empty_gold():
    raw = json.dumps(
        {"role_failed": False, "critique": "Fine.", "gold_reply": "ignored"}
    )
    out = rr._parse_review_json(raw)
    assert out is not None
    assert out.role_failed is False
    assert out.gold_reply == ""


@pytest.mark.unit
def test_parse_review_json_invalid():
    assert rr._parse_review_json("not json") is None
    assert rr._parse_review_json('{"role_failed": true}') is None


@pytest.mark.unit
def test_review_turn_skips_short_reply(monkeypatch):
    import src.config as cfg

    monkeypatch.setattr(cfg.settings, "role_review_min_reply_chars", 100, raising=False)
    assert rr.review_turn(user_text="hi", agent_reply="short") is None


@pytest.mark.unit
def test_role_review_due_every_n(monkeypatch):
    import src.config as cfg
    from src.graph.role_review_runner import role_review_due

    monkeypatch.setattr(cfg.settings, "role_review", False, raising=False)
    assert role_review_due({"turn_count": 3}) is False
    monkeypatch.setattr(cfg.settings, "role_review", True, raising=False)
    monkeypatch.setattr(cfg.settings, "role_review_every_n", 3, raising=False)
    assert role_review_due({"turn_count": 3}) is True
    assert role_review_due({"turn_count": 2}) is False


@pytest.mark.unit
def test_spawn_role_review_starts_thread(monkeypatch):
    started: list[str] = []

    class _Thread:
        def __init__(self, target=None, name=None, daemon=None):
            self._target = target
            started.append(name)

        def start(self):
            if self._target:
                self._target()

    monkeypatch.setattr("src.graph.bg_dedup.threading.Thread", _Thread)
    calls: list[str] = []
    monkeypatch.setattr(
        rrr,
        "run_role_review_for_thread",
        lambda *, thread_id: calls.append(thread_id) or 1.0,
    )
    rrr.spawn_role_review(thread_id="t1", pending=True)
    assert started == ["deferred-role-review-t1"]
    assert calls == ["t1"]


@pytest.mark.unit
def test_run_role_review_throttle_skip_direct(monkeypatch):
    from src.config import settings

    monkeypatch.setattr(settings, "role_review", True, raising=False)
    monkeypatch.setattr(settings, "role_review_every_n", 3, raising=False)
    app = MagicMock()
    app.get_state.return_value = MagicMock(
        values={
            "messages": [],
            "turn_count": 2,
        }
    )
    monkeypatch.setattr("src.agent_runner.get_graph_app", lambda: app)
    monkeypatch.setattr(
        "src.agent_runner.graph_invoke_config",
        lambda **kw: {"configurable": {"thread_id": "t1"}},
    )
    reviewed: list[str] = []
    monkeypatch.setattr(
        rrr,
        "review_turn",
        lambda **kw: reviewed.append("called") or None,
    )
    assert rrr.run_role_review_for_thread(thread_id="t1") is None
    assert reviewed == []


@pytest.mark.unit
def test_run_role_review_gold_persisted(monkeypatch):
    from langchain_core.messages import AIMessage, HumanMessage

    from src.config import settings

    monkeypatch.setattr(settings, "role_review", True, raising=False)
    app = MagicMock()
    app.get_state.return_value = MagicMock(
        values={
            "messages": [
                HumanMessage(content="Who are you?"),
                AIMessage(content="I am a helpful assistant."),
            ],
            "turn_count": 3,
        }
    )
    monkeypatch.setattr(settings, "role_review_every_n", 3, raising=False)
    monkeypatch.setattr("src.agent_runner.get_graph_app", lambda: app)
    monkeypatch.setattr(
        "src.agent_runner.graph_invoke_config",
        lambda **kw: {"configurable": {"thread_id": "t1"}},
    )
    monkeypatch.setattr(
        rrr,
        "review_turn",
        lambda **_: rr.RoleReviewOutcome(
            role_failed=True,
            critique="Assistant leak.",
            gold_reply="AGENT cold prose.",
            raw="{}",
        ),
    )
    inserted: list[dict] = []
    monkeypatch.setattr(
        rrr,
        "insert_gold_pair",
        lambda **kw: inserted.append(kw) or "uuid-1",
    )
    ms = rrr.run_role_review_for_thread(thread_id="t1")
    assert ms is not None
    assert inserted[0]["chosen_reply"] == "AGENT cold prose."
