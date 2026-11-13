"""Unit tests for shared agent runner."""

from __future__ import annotations

import asyncio
import threading
import time
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError

import src.agent_runner as ar


@pytest.mark.unit
def test_run_agent_turn_empty():
    out = ar.run_agent_turn_detailed("  ", thread_id="t")
    assert out.reply == ""
    assert out.tools_used == ()
    assert out.elapsed_ms == 0.0
    assert out.node_timings == {}


@pytest.mark.unit
def test_run_agent_turn_emits_dialogue_events(monkeypatch, caplog):
    import logging

    fake_app = MagicMock()
    fake_app.invoke.return_value = {
        "messages": [HumanMessage(content="hi"), AIMessage(content="hello back")],
    }
    fake_app.get_state.return_value = MagicMock(values={})
    monkeypatch.setattr(ar, "get_graph_app", lambda: fake_app)
    monkeypatch.setattr(ar, "session_summary_due", lambda _s: False)
    monkeypatch.setattr(
        "src.graph.reflect_runner.reflect_due",
        lambda _s: False,
    )
    events: list[str] = []

    def _capture(event: str, **fields):
        events.append(event)
        if event == "turn.user":
            assert fields.get("text") == "hi"
            assert fields.get("channel")
        if event == "turn.agent":
            assert fields.get("text") == "hello back"
        if event == "turn.complete":
            assert "node_timings" in fields

    monkeypatch.setattr(ar, "log_event", _capture)
    with caplog.at_level(logging.INFO):
        out = ar.run_agent_turn_detailed("hi", thread_id="telegram_9")
    assert out.reply == "hello back"
    assert "turn.user" in events
    assert "turn.agent" in events
    assert "turn.complete" in events
    assert (
        events.index("turn.user")
        < events.index("turn.agent")
        < events.index("turn.complete")
    )


@pytest.mark.unit
def test_run_agent_turn_inline_summary(monkeypatch):
    from src.config import settings

    fake_app = MagicMock()
    fake_app.invoke.return_value = {
        "messages": [HumanMessage(content="hi"), AIMessage(content="ok")],
    }
    fake_app.get_state.return_value = MagicMock(
        values={"turn_count": 8, "messages": []}
    )
    monkeypatch.setattr(ar, "get_graph_app", lambda: fake_app)
    monkeypatch.setattr(settings, "summary_defer", False, raising=False)
    monkeypatch.setattr(
        "src.graph.session_summary_runner.run_session_summary_if_due",
        lambda **k: 12_000.0,
    )
    monkeypatch.setattr(
        ar,
        "session_summary_due",
        lambda _s: True,
    )
    out = ar.run_agent_turn_detailed("hi", thread_id="t")
    assert out.node_timings.get("summarize_session") == 12_000.0
    assert out.summary_pending is False


@pytest.mark.unit
def test_run_agent_turn_waits_for_llm_semaphore(monkeypatch):
    held = threading.Semaphore(1)
    held.acquire()
    monkeypatch.setattr(ar, "_get_llm_turn_semaphore", lambda: held)
    fake_app = MagicMock()
    fake_app.invoke.return_value = {
        "messages": [HumanMessage(content="hi"), AIMessage(content="ok")],
    }
    monkeypatch.setattr(ar, "get_graph_app", lambda: fake_app)
    done: list[str] = []

    def _run():
        out = ar.run_agent_turn_detailed("hi", thread_id="t-sem")
        done.append(out.reply)

    worker = threading.Thread(target=_run)
    worker.start()
    time.sleep(0.05)
    assert not done
    held.release()
    worker.join(timeout=3.0)
    assert done == ["ok"]


@pytest.mark.unit
def test_run_agent_turn_waits_for_summary_lock(monkeypatch):
    held = threading.Lock()
    held.acquire()
    monkeypatch.setattr(
        "src.graph.session_summary_runner.thread_summary_lock",
        lambda _t: held,
    )
    fake_app = MagicMock()
    fake_app.invoke.return_value = {
        "messages": [HumanMessage(content="hi"), AIMessage(content="ok")],
    }
    monkeypatch.setattr(ar, "get_graph_app", lambda: fake_app)
    done: list[str] = []

    def _run():
        out = ar.run_agent_turn_detailed("hi", thread_id="t-lock")
        done.append(out.reply)

    worker = threading.Thread(target=_run)
    worker.start()
    time.sleep(0.05)
    assert not done
    held.release()
    worker.join(timeout=3.0)
    assert done == ["ok"]


@pytest.mark.unit
def test_run_agent_turn_empty_reply_fallback(monkeypatch):
    fake_app = MagicMock()
    fake_app.invoke.return_value = {
        "messages": [
            HumanMessage(content="hello"),
            AIMessage(
                content="",
                tool_calls=[{"name": "search_memory", "id": "1", "args": {}}],
            ),
        ],
    }
    monkeypatch.setattr(ar, "get_graph_app", lambda: fake_app)
    out = ar.run_agent_turn_detailed("hello", thread_id="telegram_1")
    assert "Operator:" in out.reply
    assert "No visible model output" in out.reply
    assert "Turn error" not in out.reply
    assert out.reply_is_fallback is True
    assert out.role_review_pending is False


@pytest.mark.unit
def test_run_agent_turn_happy(monkeypatch):
    fake_app = MagicMock()
    fake_app.invoke.return_value = {
        "messages": [
            HumanMessage(content="hi"),
            ToolMessage(content="ok", tool_call_id="1", name="search_web"),
            AIMessage(content="Shalom."),
        ],
        "retrieved_memories": [],
    }
    monkeypatch.setattr(ar, "get_graph_app", lambda: fake_app)
    out = ar.run_agent_turn_detailed("hi", thread_id="test-thread")
    assert out.reply == "Shalom."
    assert out.tools_used == ("search_web",)
    assert out.elapsed_ms >= 0.0
    cfg = fake_app.invoke.call_args.kwargs["config"]
    assert cfg["configurable"]["thread_id"] == "test-thread"


@pytest.mark.unit
def test_run_agent_turn_counts_prefetch_search(monkeypatch):
    fake_app = MagicMock()
    fake_app.invoke.return_value = {
        "messages": [HumanMessage(content="google x"), AIMessage(content="ok")],
        "retrieved_memories": [
            "[WEB] search_web: query='x'\nWeb search: x\n\n1. Hit\n   snippet"
        ],
    }
    monkeypatch.setattr(ar, "get_graph_app", lambda: fake_app)
    out = ar.run_agent_turn_detailed("google x", thread_id="t")
    assert out.tools_used == ("search_web",)


@pytest.mark.unit
def test_run_agent_turn_recursion(monkeypatch):
    fake_app = MagicMock()
    fake_app.invoke.side_effect = GraphRecursionError("limit")
    monkeypatch.setattr(ar, "get_graph_app", lambda: fake_app)
    out = ar.run_agent_turn("hi", thread_id="t")
    assert "Graph step limit" in out


@pytest.mark.unit
def test_run_agent_turn_exception(monkeypatch):
    fake_app = MagicMock()
    fake_app.invoke.side_effect = RuntimeError("boom")
    monkeypatch.setattr(ar, "get_graph_app", lambda: fake_app)
    out = ar.run_agent_turn("hi", thread_id="t")
    assert "Internal error" in out


@pytest.mark.unit
def test_arun_agent_turn(monkeypatch):
    monkeypatch.setattr(
        ar,
        "run_agent_turn_detailed",
        lambda *a, **k: ar.AgentTurnResult(reply="ok", tools_used=(), elapsed_ms=1.0),
    )
    result = asyncio.run(ar.arun_agent_turn("x", thread_id="t"))
    assert result.reply == "ok"


@pytest.mark.unit
def test_run_agent_turn_invoke_payload(monkeypatch):
    fake_app = MagicMock()
    fake_app.invoke.return_value = {"messages": [AIMessage(content="ok")]}
    monkeypatch.setattr(ar, "get_graph_app", lambda: fake_app)
    ar.run_agent_turn("hi", thread_id="t", session_uuid="telegram-1")
    payload = fake_app.invoke.call_args.args[0]
    assert set(payload.keys()) == {"messages", "current_time", "session_uuid"}
    assert payload["session_uuid"] == "telegram-1"


@pytest.mark.unit
def test_run_agent_turn_summary_pending(monkeypatch):
    from src.config import settings

    fake_app = MagicMock()
    fake_app.invoke.return_value = {
        "messages": [HumanMessage(content="hi"), AIMessage(content="ok")],
    }
    fake_app.get_state.return_value = MagicMock(
        values={"turn_count": 8, "messages": []}
    )
    monkeypatch.setattr(ar, "get_graph_app", lambda: fake_app)
    monkeypatch.setattr(settings, "summary_defer", True, raising=False)
    monkeypatch.setattr(ar, "session_summary_due", lambda _s: True)
    out = ar.run_agent_turn_detailed("hi", thread_id="telegram_1")
    assert out.summary_pending is True


@pytest.mark.unit
def test_run_agent_turn_reflect_pending(monkeypatch):
    from src.config import settings

    fake_app = MagicMock()
    fake_app.invoke.return_value = {
        "messages": [HumanMessage(content="hi"), AIMessage(content="ok")],
    }
    fake_app.get_state.return_value = MagicMock(
        values={
            "messages": [HumanMessage(content="hi"), AIMessage(content="ok")],
            "needs_reflect": True,
            "turn_count": 3,
        }
    )
    monkeypatch.setattr(ar, "get_graph_app", lambda: fake_app)
    monkeypatch.setattr(settings, "reflect_defer", True, raising=False)
    monkeypatch.setattr(settings, "reflection", True, raising=False)
    monkeypatch.setattr("src.graph.reflect_runner.reflect_due", lambda _s: True)
    out = ar.run_agent_turn_detailed("hi", thread_id="telegram_1")
    assert out.reflect_pending is True


@pytest.mark.unit
def test_run_agent_turn_role_review_pending(monkeypatch):
    from src.config import settings

    fake_app = MagicMock()
    fake_app.invoke.return_value = {
        "messages": [
            HumanMessage(content="hi"),
            AIMessage(content="ok reply long enough"),
        ],
    }
    fake_app.get_state.return_value = MagicMock(
        values={
            "messages": [
                HumanMessage(content="hi"),
                AIMessage(content="ok reply long enough"),
            ],
            "turn_count": 3,
        }
    )
    monkeypatch.setattr(ar, "get_graph_app", lambda: fake_app)
    monkeypatch.setattr(settings, "role_review", True, raising=False)
    monkeypatch.setattr(settings, "role_review_every_n", 3, raising=False)
    monkeypatch.setattr("src.graph.reflect_runner.reflect_due", lambda _s: False)
    out = ar.run_agent_turn_detailed("hi", thread_id="telegram_1")
    assert out.role_review_pending is True


@pytest.mark.unit
def test_run_agent_turn_role_review_throttled(monkeypatch):
    from src.config import settings

    fake_app = MagicMock()
    fake_app.invoke.return_value = {
        "messages": [
            HumanMessage(content="hi"),
            AIMessage(content="ok reply long enough"),
        ],
    }
    fake_app.get_state.return_value = MagicMock(
        values={
            "messages": [
                HumanMessage(content="hi"),
                AIMessage(content="ok reply long enough"),
            ],
            "turn_count": 2,
        }
    )
    monkeypatch.setattr(ar, "get_graph_app", lambda: fake_app)
    monkeypatch.setattr(settings, "role_review", True, raising=False)
    monkeypatch.setattr(settings, "role_review_every_n", 3, raising=False)
    out = ar.run_agent_turn_detailed("hi", thread_id="telegram_1")
    assert out.role_review_pending is False


@pytest.mark.unit
def test_run_agent_turn_reflect_inline(monkeypatch):
    from src.config import settings

    fake_app = MagicMock()
    fake_app.invoke.return_value = {
        "messages": [HumanMessage(content="hi"), AIMessage(content="ok")],
    }
    fake_app.get_state.return_value = MagicMock(
        values={
            "messages": [HumanMessage(content="hi"), AIMessage(content="ok")],
            "needs_reflect": True,
            "turn_count": 3,
        }
    )
    monkeypatch.setattr(ar, "get_graph_app", lambda: fake_app)
    monkeypatch.setattr(settings, "reflect_defer", False, raising=False)
    monkeypatch.setattr(settings, "reflection", True, raising=False)
    out = ar.run_agent_turn_detailed("hi", thread_id="telegram_1")
    assert out.reflect_pending is False
    assert "reflect_node" not in (out.node_timings or {})


@pytest.mark.unit
def test_infer_reply_channel():
    assert ar.infer_reply_channel("telegram_42") == "telegram"
    assert ar.infer_reply_channel("cli-session") is None


@pytest.mark.unit
def test_run_agent_turn_telegram_channel(monkeypatch):
    fake_app = MagicMock()
    fake_app.invoke.return_value = {"messages": [AIMessage(content="ok")]}
    monkeypatch.setattr(ar, "get_graph_app", lambda: fake_app)
    ar.run_agent_turn("hi", thread_id="telegram_99")
    cfg = fake_app.invoke.call_args.kwargs["config"]
    assert cfg["configurable"]["reply_channel"] == "telegram"


@pytest.mark.unit
def test_cli_reply_strips_think_tags(monkeypatch):
    fake_app = MagicMock()
    fake_app.invoke.return_value = {
        "messages": [
            AIMessage(content="<think>plan</think>\n\nSpoken cut."),
        ]
    }
    fake_app.get_state.return_value = MagicMock(values={})
    monkeypatch.setattr(ar, "get_graph_app", lambda: fake_app)
    monkeypatch.setattr(ar, "session_summary_due", lambda _s: False)
    monkeypatch.setattr("src.graph.reflect_runner.reflect_due", lambda _s: False)
    out = ar.run_agent_turn_detailed("hi", thread_id="cli-session")
    assert "<think>" not in out.reply
    assert "Spoken cut." in out.reply


@pytest.mark.unit
def test_telegram_reply_keeps_think_for_blockquote(monkeypatch):
    fake_app = MagicMock()
    fake_app.invoke.return_value = {
        "messages": [
            AIMessage(content="<think>plan</think>\n\nSpoken cut."),
        ]
    }
    fake_app.get_state.return_value = MagicMock(values={})
    monkeypatch.setattr(ar, "get_graph_app", lambda: fake_app)
    monkeypatch.setattr(ar, "session_summary_due", lambda _s: False)
    monkeypatch.setattr("src.graph.reflect_runner.reflect_due", lambda _s: False)
    out = ar.run_agent_turn_detailed("hi", thread_id="telegram_99")
    assert "<think>" in out.reply
    assert "Spoken cut." in out.reply


@pytest.mark.unit
def test_reset_graph_app():
    ar._graph_app = object()
    ar.reset_graph_app()
    assert ar._graph_app is None


@pytest.mark.unit
def test_checkpoint_state_empty_values(monkeypatch):
    fake_app = MagicMock()
    fake_app.invoke.return_value = {"messages": [AIMessage(content="ok")]}
    fake_app.get_state.return_value = MagicMock(values=None)
    monkeypatch.setattr(ar, "get_graph_app", lambda: fake_app)
    out = ar.run_agent_turn_detailed("hi", thread_id="t")
    assert out.summary_pending is False


@pytest.mark.unit
def test_checkpoint_state_read_error(monkeypatch):
    fake_app = MagicMock()
    fake_app.invoke.return_value = {"messages": [AIMessage(content="ok")]}
    fake_app.get_state.side_effect = RuntimeError("no checkpoint")
    monkeypatch.setattr(ar, "get_graph_app", lambda: fake_app)
    out = ar.run_agent_turn_detailed("hi", thread_id="t")
    assert out.reply == "ok"
    assert out.summary_pending is False
