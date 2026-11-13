"""Per-turn tool round cap and recursion settings."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

import src.graph.workflow as w
from src.graph.time_node import update_current_time
from src.graph.tool_round_limit import (
    merge_tool_round_increment,
    tool_round_cap_reached,
)


def test_tool_round_cap_reached_respects_config(monkeypatch):
    from src.config import settings

    monkeypatch.setattr(settings, "max_tool_rounds_per_turn", 2, raising=False)
    assert not tool_round_cap_reached({"tool_rounds": 1})
    assert tool_round_cap_reached({"tool_rounds": 2})
    monkeypatch.setattr(settings, "max_tool_rounds_per_turn", 0, raising=False)
    assert not tool_round_cap_reached({"tool_rounds": 99})


def test_merge_tool_round_increment():
    state = {"tool_rounds": 1}
    out = merge_tool_round_increment(state, {"messages": []})
    assert out["tool_rounds"] == 2


@pytest.mark.unit
def test_merge_tool_round_increment_non_dict_out():
    state = {"tool_rounds": 4}
    out = merge_tool_round_increment(state, None)
    assert out["tool_rounds"] == 5
    # Every round also appends a ledger entry — the loop's memory of itself,
    # kept in state because the message window gets trimmed.
    assert len(out["tool_ledger"]) == 1
    assert out["tool_ledger"][0]["round"] == 5
    assert out["tool_ledger"][0]["fruitless"] is True


def test_update_time_resets_tool_rounds():
    state = {
        "messages": [HumanMessage(content="hi")],
        "tool_rounds": 3,
        "turn_count": 4,
    }
    out = update_current_time(state)  # type: ignore[arg-type]
    assert out.get("tool_rounds") == 0
    assert out.get("turn_count") == 5


def test_route_agent_blocks_tools_at_cap(monkeypatch):
    from src.config import settings

    monkeypatch.setattr(settings, "max_tool_rounds_per_turn", 1, raising=False)
    monkeypatch.setattr(settings, "reflection", False, raising=False)
    state = {
        "messages": [
            HumanMessage(content="q"),
            AIMessage(
                content="",
                tool_calls=[{"name": "search_memory", "id": "c1", "args": {}}],
            ),
        ],
        "tool_rounds": 1,
        "needs_reflect": False,
    }
    assert w._route_agent_after_llm(state) == "tool_cap_cleanup"


@pytest.mark.unit
@pytest.mark.parametrize(
    "messages",
    [
        [],
        [HumanMessage(content="x")],
        [AIMessage(content="plain")],
    ],
)
def test_tool_cap_cleanup_node_noop_without_pending_tools(messages):
    state = {"messages": messages} if messages else {}
    assert w.tool_cap_cleanup_node(state) == {}


@pytest.mark.unit
def test_has_pending_tool_calls_requires_trailing_ai_with_tools():
    assert w._has_pending_tool_calls({}) is False
    assert w._has_pending_tool_calls({"messages": [HumanMessage(content="x")]}) is False
    assert (
        w._has_pending_tool_calls(
            {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[{"name": "search_memory", "id": "c1", "args": {}}],
                    )
                ]
            }
        )
        is True
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "reflection,reflect_every_n,turn_count,expected",
    [
        (False, 1, 1, "end"),
        (True, 1, 1, "reflect"),
    ],
)
def test_route_agent_at_cap_without_pending_tools(
    monkeypatch, reflection, reflect_every_n, turn_count, expected
):
    from src.config import settings

    monkeypatch.setattr(settings, "max_tool_rounds_per_turn", 1, raising=False)
    monkeypatch.setattr(settings, "reflection", reflection, raising=False)
    monkeypatch.setattr(settings, "reflect_every_n", reflect_every_n, raising=False)
    monkeypatch.setattr(settings, "reflect_defer", False, raising=False)
    monkeypatch.setattr(w, "tools_condition", lambda _s: "tools", raising=True)
    state = {
        "messages": [HumanMessage(content="x")],
        "tool_rounds": 1,
        "needs_reflect": True,
        "turn_count": turn_count,
    }
    assert w._route_agent_after_llm(state) == expected


def test_route_agent_allows_tools_under_cap(monkeypatch):
    from src.config import settings

    monkeypatch.setattr(settings, "max_tool_rounds_per_turn", 2, raising=False)
    state = {
        "messages": [
            HumanMessage(content="q"),
            AIMessage(
                content="",
                tool_calls=[{"name": "search_memory", "id": "c1", "args": {}}],
            ),
        ],
        "tool_rounds": 0,
    }
    assert w._route_agent_after_llm(state) == "tools"


def test_route_agent_without_tools_uses_reflect_route(monkeypatch):
    monkeypatch.setattr(w, "tools_condition", lambda _s: "__end__", raising=True)
    monkeypatch.setattr(w, "_reflect_route_or_end", lambda _s: "end", raising=True)
    assert w._route_agent_after_llm({"messages": []}) == "end"


def test_tools_node_increments_tool_rounds(monkeypatch):
    monkeypatch.setattr(
        w,
        "invoke_tool_node_logged",
        lambda _node, _state, _config: {"messages": []},
        raising=True,
    )
    out = w.tools_node({"tool_rounds": 0, "messages": []}, {})  # type: ignore[arg-type]
    assert out["tool_rounds"] == 1


def test_agent_node_disables_tools_when_round_cap_reached(monkeypatch):
    from src.config import settings

    monkeypatch.setattr(settings, "max_tool_rounds_per_turn", 1, raising=False)

    called = {"plain": 0, "tools": 0}

    class _PlainLLM:
        def invoke(self, _msgs, **_kwargs):
            called["plain"] += 1
            return AIMessage(content="answer without tools")

    class _ToolLLM:
        def invoke(self, _msgs, **_kwargs):
            called["tools"] += 1
            return AIMessage(
                content="",
                tool_calls=[{"name": "search_memory", "id": "c1", "args": {}}],
            )

    from tests.support.llm_mocks import patch_agent_llm_split

    patch_agent_llm_split(monkeypatch, _PlainLLM(), _ToolLLM())

    state = {
        "messages": [HumanMessage(content="first turn question about memory")],
        "retrieved_memories": [],
        "current_time": "",
        "session_summary": "",
        "tool_rounds": 1,
    }
    w.agent_node(state)  # type: ignore[arg-type]
    assert called == {"plain": 1, "tools": 0}
