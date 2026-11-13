"""Tests for deferred reflect runner."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.graph import reflect_runner as rr


@pytest.mark.unit
def test_run_reflect_if_due_no_thread_id():
    assert rr.run_reflect_if_due(thread_id="  ") is None


@pytest.mark.unit
def test_run_reflect_if_due_skips(monkeypatch):
    app = MagicMock()
    app.get_state.return_value = MagicMock(
        values={
            "messages": [HumanMessage(content="x"), AIMessage(content="y")],
            "needs_reflect": True,
            "turn_count": 1,
        }
    )
    monkeypatch.setattr(rr, "get_graph_app", lambda: app)
    monkeypatch.setattr(rr, "reflect_due", lambda _s: False)
    assert rr.run_reflect_if_due(thread_id="t1") is None


@pytest.mark.unit
def test_run_reflect_if_due_success(monkeypatch):
    app = MagicMock()
    app.get_state.return_value = MagicMock(
        values={
            "messages": [HumanMessage(content="x"), AIMessage(content="y")],
            "needs_reflect": True,
            "turn_count": 3,
        }
    )
    monkeypatch.setattr(rr, "get_graph_app", lambda: app)
    monkeypatch.setattr(rr, "reflect_due", lambda _s: True)
    monkeypatch.setattr(
        "src.graph.workflow.reflect_node",
        lambda _s: {"last_reflect_turn": 3},
    )
    ms = rr.run_reflect_if_due(thread_id="t1")
    assert ms is not None and ms >= 0
    app.update_state.assert_called_once()


@pytest.mark.unit
def test_spawn_reflect_if_pending_starts_thread(monkeypatch):
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
        rr,
        "run_reflect_if_due",
        lambda *, thread_id: calls.append(thread_id) or 1.0,
    )
    rr.spawn_reflect_if_pending(thread_id="t1", reflect_pending=True)
    assert started == ["deferred-reflect-t1"]
    assert calls == ["t1"]


@pytest.mark.unit
def test_spawn_reflect_noop():
    rr.spawn_reflect_if_pending(thread_id="t1", reflect_pending=False)


@pytest.mark.unit
def test_schedule_reflect_background_defer(monkeypatch):
    monkeypatch.setattr("src.config.settings.reflect_defer", True)
    app = MagicMock()
    app.get_state.return_value = MagicMock(
        values={"needs_reflect": True, "turn_count": 3}
    )
    monkeypatch.setattr(rr, "get_graph_app", lambda: app)
    monkeypatch.setattr(rr, "reflect_due", lambda _s: True)
    monkeypatch.setattr(rr, "spawn_reflect_if_pending", lambda **k: None)
    assert rr.schedule_reflect_background(thread_id="t1") is True


@pytest.mark.unit
def test_run_reflect_if_due_reflect_node_error(monkeypatch):
    app = MagicMock()
    app.get_state.return_value = MagicMock(
        values={"needs_reflect": True, "turn_count": 3}
    )
    monkeypatch.setattr(rr, "get_graph_app", lambda: app)
    monkeypatch.setattr(rr, "reflect_due", lambda _s: True)

    def _boom(_s):
        raise RuntimeError("fail")

    monkeypatch.setattr("src.graph.workflow.reflect_node", _boom)
    assert rr.run_reflect_if_due(thread_id="t1") is None


@pytest.mark.unit
def test_run_reflect_if_due_empty_result(monkeypatch):
    app = MagicMock()
    app.get_state.return_value = MagicMock(
        values={"needs_reflect": True, "turn_count": 3}
    )
    monkeypatch.setattr(rr, "get_graph_app", lambda: app)
    monkeypatch.setattr(rr, "reflect_due", lambda _s: True)
    monkeypatch.setattr("src.graph.workflow.reflect_node", lambda _s: {})
    assert rr.run_reflect_if_due(thread_id="t1") is None


@pytest.mark.unit
def test_run_reflect_if_due_get_state_error(monkeypatch):
    app = MagicMock()
    app.get_state.side_effect = RuntimeError("db")
    monkeypatch.setattr(rr, "get_graph_app", lambda: app)
    assert rr.run_reflect_if_due(thread_id="t1") is None


@pytest.mark.unit
def test_run_reflect_if_due_no_values(monkeypatch):
    app = MagicMock()
    app.get_state.return_value = MagicMock(values=None)
    monkeypatch.setattr(rr, "get_graph_app", lambda: app)
    assert rr.run_reflect_if_due(thread_id="t1") is None


@pytest.mark.unit
def test_run_reflect_if_due_update_state_error(monkeypatch):
    app = MagicMock()
    app.get_state.return_value = MagicMock(
        values={"needs_reflect": True, "turn_count": 3}
    )
    app.update_state.side_effect = RuntimeError("write fail")
    monkeypatch.setattr(rr, "get_graph_app", lambda: app)
    monkeypatch.setattr(rr, "reflect_due", lambda _s: True)
    monkeypatch.setattr(
        "src.graph.workflow.reflect_node",
        lambda _s: {"last_reflect_turn": 3},
    )
    assert rr.run_reflect_if_due(thread_id="t1") is None


@pytest.mark.unit
def test_spawn_reflect_empty_thread():
    rr.spawn_reflect_if_pending(thread_id="  ", reflect_pending=True)


@pytest.mark.unit
def test_schedule_reflect_background_empty_thread():
    assert rr.schedule_reflect_background(thread_id="") is False


@pytest.mark.unit
def test_schedule_reflect_background_get_state_error(monkeypatch):
    monkeypatch.setattr("src.config.settings.reflect_defer", True)
    app = MagicMock()
    app.get_state.side_effect = RuntimeError("nope")
    monkeypatch.setattr(rr, "get_graph_app", lambda: app)
    assert rr.schedule_reflect_background(thread_id="t1") is False


@pytest.mark.unit
def test_reflect_due_delegates(monkeypatch):
    monkeypatch.setattr(
        "src.graph.workflow.reflect_eligible",
        lambda _s: True,
    )
    assert rr.reflect_due({}) is True
