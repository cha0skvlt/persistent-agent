"""Tests for deferred session summary runner."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.graph import session_summary_runner as ssr


@pytest.mark.unit
def test_run_session_summary_if_due_no_thread_id():
    assert ssr.run_session_summary_if_due(thread_id="  ") is None


@pytest.mark.unit
def test_thread_summary_lock_empty_thread_id(caplog):
    import logging

    with caplog.at_level(logging.WARNING):
        with ssr.thread_summary_lock("  ") as lock:
            assert lock is ssr._NOOP_LOCK
    assert "empty thread_id" in caplog.text


@pytest.mark.unit
def test_run_session_summary_if_due_skips(monkeypatch):
    app = MagicMock()
    app.get_state.return_value = MagicMock(values={"turn_count": 1})
    monkeypatch.setattr(ssr, "get_graph_app", lambda: app)
    monkeypatch.setattr(ssr, "session_summary_due", lambda _s: False)
    assert ssr.run_session_summary_if_due(thread_id="t1") is None


@pytest.mark.unit
def test_run_session_summary_if_due_no_state(monkeypatch):
    app = MagicMock()
    app.get_state.return_value = None
    monkeypatch.setattr(ssr, "get_graph_app", lambda: app)
    assert ssr.run_session_summary_if_due(thread_id="t1") is None


@pytest.mark.unit
def test_run_session_summary_if_due_get_state_error(monkeypatch):
    app = MagicMock()
    app.get_state.side_effect = RuntimeError("db down")
    monkeypatch.setattr(ssr, "get_graph_app", lambda: app)
    assert ssr.run_session_summary_if_due(thread_id="t1") is None


@pytest.mark.unit
def test_run_session_summary_if_due_summarize_error(monkeypatch):
    app = MagicMock()
    app.get_state.return_value = MagicMock(values={"turn_count": 8})
    monkeypatch.setattr(ssr, "get_graph_app", lambda: app)
    monkeypatch.setattr(ssr, "session_summary_due", lambda _s: True)

    def _boom(*_a, **_k):
        raise RuntimeError("llm fail")

    monkeypatch.setattr(ssr, "summarize_session", _boom)
    assert ssr.run_session_summary_if_due(thread_id="t1") is None


@pytest.mark.unit
def test_run_session_summary_if_due_empty_result(monkeypatch):
    app = MagicMock()
    app.get_state.return_value = MagicMock(values={"turn_count": 8})
    monkeypatch.setattr(ssr, "get_graph_app", lambda: app)
    monkeypatch.setattr(ssr, "session_summary_due", lambda _s: True)
    monkeypatch.setattr(ssr, "summarize_session", lambda *_a, **_k: {})
    assert ssr.run_session_summary_if_due(thread_id="t1") is None


@pytest.mark.unit
def test_run_session_summary_if_due_update_state_error(monkeypatch):
    app = MagicMock()
    app.get_state.return_value = MagicMock(
        values={
            "turn_count": 8,
            "messages": [HumanMessage(content="hi"), AIMessage(content="ok")],
        }
    )
    app.update_state.side_effect = RuntimeError("write fail")
    monkeypatch.setattr(ssr, "get_graph_app", lambda: app)
    monkeypatch.setattr(ssr, "session_summary_due", lambda _s: True)
    monkeypatch.setattr(
        ssr,
        "summarize_session",
        lambda *_a, **_k: {"session_summary": "compressed", "last_summary_turn": 8},
    )
    assert ssr.run_session_summary_if_due(thread_id="telegram_1") is None


@pytest.mark.unit
def test_run_session_summary_if_due_updates_state(monkeypatch):
    app = MagicMock()
    app.get_state.return_value = MagicMock(
        values={
            "turn_count": 8,
            "messages": [HumanMessage(content="hi"), AIMessage(content="ok")],
        }
    )
    monkeypatch.setattr(ssr, "get_graph_app", lambda: app)
    monkeypatch.setattr(ssr, "session_summary_due", lambda _s: True)
    monkeypatch.setattr(
        ssr,
        "summarize_session",
        lambda *_a, **_k: {"session_summary": "compressed", "last_summary_turn": 8},
    )
    ms = ssr.run_session_summary_if_due(thread_id="telegram_1")
    assert ms is not None and ms >= 0
    app.update_state.assert_called_once()


@pytest.mark.unit
def test_spawn_session_summary_if_pending(monkeypatch):
    started: list[str] = []

    class _Thread:
        def __init__(self, target=None, name=None, daemon=None):
            self._target = target
            started.append(name)

        def start(self):
            if self._target:
                self._target()

    monkeypatch.setattr(ssr.threading, "Thread", _Thread)
    calls: list[str] = []
    monkeypatch.setattr(
        ssr,
        "run_session_summary_if_due",
        lambda **k: calls.append(k["thread_id"]),
    )
    ssr.spawn_session_summary_if_pending(thread_id="", summary_pending=True)
    ssr.spawn_session_summary_if_pending(thread_id="t1", summary_pending=False)
    ssr.spawn_session_summary_if_pending(thread_id="t1", summary_pending=True)
    assert started == ["deferred-summary-t1"]
    assert calls == ["t1"]


@pytest.mark.unit
def test_schedule_session_summary_background(monkeypatch):
    app = MagicMock()
    app.get_state.return_value = MagicMock(values={"turn_count": 8})
    monkeypatch.setattr(ssr, "get_graph_app", lambda: app)
    monkeypatch.setattr(ssr, "session_summary_due", lambda _s: True)
    monkeypatch.setattr(ssr, "run_session_summary_if_due", lambda **k: None)
    started: list = []

    class _Thread:
        def __init__(self, target=None, name=None, daemon=None):
            started.append(name)
            self._target = target

        def start(self):
            if self._target:
                self._target()

    monkeypatch.setattr(ssr.threading, "Thread", _Thread)
    assert ssr.schedule_session_summary_background(thread_id="repl") is True
    assert started


@pytest.mark.unit
def test_schedule_session_summary_background_inline(monkeypatch):
    from src.config import settings

    monkeypatch.setattr(settings, "summary_defer", False, raising=False)
    calls: list[str] = []

    monkeypatch.setattr(
        ssr,
        "run_session_summary_if_due",
        lambda **k: calls.append(k["thread_id"]) or None,
    )
    assert ssr.schedule_session_summary_background(thread_id="repl") is False
    assert calls == ["repl"]


@pytest.mark.unit
def test_schedule_session_summary_background_state_error(monkeypatch):
    app = MagicMock()
    app.get_state.side_effect = RuntimeError("no state")
    monkeypatch.setattr(ssr, "get_graph_app", lambda: app)
    assert ssr.schedule_session_summary_background(thread_id="repl") is False
