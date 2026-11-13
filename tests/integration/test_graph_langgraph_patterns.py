"""LangGraph official testing patterns — MemorySaver, node invoke, partial paths, eval.

See https://docs.langchain.com/oss/python/langgraph/test
"""

from __future__ import annotations

import uuid

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.graph import workflow as w
from tests.support.graph_fixtures import (
    fresh_memory_graph,
    fresh_thread_config,
    graph_test_settings,
    initial_turn_input,
    patch_retrieve_noop,
    seed_post_retrieve_state,
)
from tests.support.helpers import graph_state


@pytest.mark.integration
def test_memory_graph_fresh_thread_ids_isolated(
    memory_graph_app,
    mock_agent_llm,
    retrieve_mocks,
):
    """Each ``thread_id`` gets its own checkpoint — no cross-test bleed."""
    mock_agent_llm("reply-a")
    cfg_a = fresh_thread_config(suffix=f"a_{uuid.uuid4().hex[:8]}")
    cfg_b = fresh_thread_config(suffix=f"b_{uuid.uuid4().hex[:8]}")

    memory_graph_app.invoke(initial_turn_input("hello A"), config=cfg_a)
    memory_graph_app.invoke(initial_turn_input("hello B"), config=cfg_b)

    snap_a = memory_graph_app.get_state(cfg_a)
    snap_b = memory_graph_app.get_state(cfg_b)
    texts_a = [getattr(m, "content", "") for m in snap_a.values.get("messages") or []]
    texts_b = [getattr(m, "content", "") for m in snap_b.values.get("messages") or []]

    assert any("hello A" in t for t in texts_a)
    assert any("hello B" in t for t in texts_b)
    assert not any("hello B" in t for t in texts_a)
    assert not any("hello A" in t for t in texts_b)


@pytest.mark.integration
def test_two_turn_checkpoint_persists_messages(
    memory_graph_app,
    graph_thread_config,
    mock_agent_llm,
    retrieve_mocks,
):
    """Second invoke on the same ``thread_id`` appends to checkpoint history."""
    mock_agent_llm("first reply")
    inp = initial_turn_input("turn one")
    memory_graph_app.invoke(inp, config=graph_thread_config)
    after_one = memory_graph_app.get_state(graph_thread_config)
    count_one = len(after_one.values.get("messages") or [])

    mock_agent_llm("second reply")
    memory_graph_app.invoke(
        {"messages": [HumanMessage(content="turn two")], "current_time": ""},
        config=graph_thread_config,
    )
    after_two = memory_graph_app.get_state(graph_thread_config)
    msgs = after_two.values.get("messages") or []

    assert len(msgs) > count_one
    assert any(getattr(m, "content", "") == "turn two" for m in msgs)
    assert any(getattr(m, "content", "") == "second reply" for m in msgs)


@pytest.mark.integration
@pytest.mark.db
def test_graph_node_agent_invoke_mock_llm(monkeypatch):
    """Isolate ``agent`` node via ``graph.nodes[...].invoke()`` (no checkpointer)."""
    graph_test_settings(monkeypatch)
    app = fresh_memory_graph()
    monkeypatch.setattr(
        w, "invoke_llm", lambda *a, **k: AIMessage(content="mock agent reply")
    )
    monkeypatch.setattr(w, "_agent_may_call_tools", lambda *a, **k: True)

    state = seed_post_retrieve_state("hello")
    out = app.nodes["agent"].invoke(state)
    last = out["messages"][-1]
    assert isinstance(last, AIMessage)
    assert last.content == "mock agent reply"


@pytest.mark.integration
def test_graph_node_retrieve_memory_invoke(monkeypatch):
    """Isolate ``retrieve_memory`` via ``graph.nodes[...].invoke()``."""
    graph_test_settings(monkeypatch)
    app = fresh_memory_graph()
    mem = patch_retrieve_noop(monkeypatch)

    state = graph_state("what is in the phylactery stack", current_time="2026-06-30")
    from src.graph.route_turn_node import route_turn_node

    state.update(route_turn_node(state))
    out = app.nodes["retrieve_memory"].invoke(state)

    assert "retrieved_memories" in out
    assert mem.search_calls
    assert mem.search_calls[-1]["query"] == "what is in the phylactery stack"


@pytest.mark.integration
@pytest.mark.db
def test_partial_invoke_interrupt_after_tools(
    memory_graph_app,
    graph_thread_config,
    mock_agent_llm,
    retrieve_mocks,
):
    """``update_state`` + ``interrupt_after='tools'`` — one tool round, no second agent."""
    mock_agent_llm(
        AIMessage(
            content="",
            tool_calls=[
                {"name": "search_memory", "id": "call-1", "args": {"query": "agent"}}
            ],
        )
    )
    seeded = seed_post_retrieve_state("search memory for agent")
    memory_graph_app.update_state(
        graph_thread_config,
        values=seeded,
        as_node="retrieve_memory",
    )

    result = memory_graph_app.invoke(
        None,
        config=graph_thread_config,
        interrupt_after="tools",
    )
    msgs = result.get("messages") or []
    types = [type(m).__name__ for m in msgs]

    assert "AIMessage" in types
    assert "ToolMessage" in types
    assert any(isinstance(m, ToolMessage) and m.name == "search_memory" for m in msgs)
    # Stopped after tools — no second agent synthesis in this partial run.
    assert types.count("AIMessage") == 1


@pytest.mark.integration
def test_full_invoke_text_only_mock_llm(
    memory_graph_app,
    graph_thread_config,
    mock_agent_llm,
    retrieve_mocks,
):
    """CI-friendly full ``invoke`` — mocked LLM, no Ollama."""
    mock_agent_llm("**Agent** online — one line reply.")
    result = memory_graph_app.invoke(
        initial_turn_input("hello"),
        config=graph_thread_config,
    )
    msgs = result.get("messages") or []
    ai_msgs = [m for m in msgs if isinstance(m, AIMessage)]
    assert ai_msgs
    assert "Agent" in (ai_msgs[-1].content or "")
    assert not getattr(ai_msgs[-1], "tool_calls", None)


@pytest.mark.integration
def test_trajectory_greeting_no_tools_regression(
    memory_graph_app,
    graph_thread_config,
    mock_agent_llm,
    retrieve_mocks,
):
    """Deterministic persona regression — greeting stays text-only (local eval)."""
    mock_agent_llm("Hello. I am Agent.")
    result = memory_graph_app.invoke(
        initial_turn_input("hello"),
        config=graph_thread_config,
    )
    msgs = result.get("messages") or []
    tool_msgs = [m for m in msgs if isinstance(m, ToolMessage)]
    ai_with_tools = [
        m
        for m in msgs
        if isinstance(m, AIMessage)
        and getattr(m, "tool_calls", None)
        and len(m.tool_calls) > 0
    ]
    assert not tool_msgs
    assert not ai_with_tools
    assert any(isinstance(m, AIMessage) and "Agent" in (m.content or "") for m in msgs)
