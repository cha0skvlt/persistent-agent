"""Graph freedom — unconditional tools→agent and round-cap brake."""

from __future__ import annotations

import uuid

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.config import settings
from src.graph import workflow as w
from tests.support.graph_fixtures import graph_test_settings, patch_retrieve_noop
from tests.support.llm_mocks import patch_agent_llm


class _ScriptedLLM:
    """Pop scripted AIMessages; last reply reused if the script runs dry."""

    def __init__(self, responses: list[AIMessage]):
        self._responses = list(responses)
        self._last: AIMessage | None = None

    def invoke(self, _msgs, **_kwargs):
        if self._responses:
            self._last = self._responses.pop(0)
            return self._last
        if self._last is not None:
            return self._last
        return AIMessage(content="done")


def _tool_ai(query: str, *, call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "search_lessons",
                "id": call_id,
                "args": {"query": query},
                "type": "tool_call",
            }
        ],
    )


def _prep_graph(monkeypatch: pytest.MonkeyPatch, script: list[AIMessage]):
    graph_test_settings(monkeypatch)
    monkeypatch.setattr(settings, "reflection", False, raising=False)
    patch_retrieve_noop(monkeypatch)
    patch_agent_llm(monkeypatch, _ScriptedLLM(script))
    return w.build_app(for_studio=True)


def _invoke(app, *, human: str = "find a lesson"):
    return app.invoke(
        {
            "messages": [HumanMessage(content=human)],
            "current_time": "",
            "session_uuid": uuid.uuid4().hex,
        },
        config={"recursion_limit": 40},
    )


@pytest.mark.integration
def test_tools_route_back_to_agent_always():
    from src.graph.workflow import after_tools_condition

    assert after_tools_condition({"messages": []}) == "agent"
    assert (
        after_tools_condition(
            {
                "messages": [
                    ToolMessage(content="x", tool_call_id="1", name="save_memory")
                ]
            }
        )
        == "agent"
    )


@pytest.mark.integration
def test_agent_repeats_tool_with_new_args(monkeypatch):
    calls = {"n": 0}

    def _search(*, query: str = "", limit: int = 3) -> str:
        calls["n"] += 1
        return f"hit:{query}"

    monkeypatch.setattr(
        "src.tools.lesson_tools.search_lessons_impl", _search, raising=True
    )
    app = _prep_graph(
        monkeypatch,
        [
            _tool_ai("a", call_id="c1"),
            _tool_ai("b", call_id="c2"),
            AIMessage(content="Slice across two lessons."),
        ],
    )
    out = _invoke(app)
    msgs = out.get("messages") or []
    assert calls["n"] == 2
    assert int(out.get("tool_rounds") or 0) == 2
    last = msgs[-1]
    assert isinstance(last, AIMessage)
    assert not getattr(last, "tool_calls", None)
    assert "Slice" in str(last.content)


@pytest.mark.integration
def test_agent_may_repeat_exact_duplicate_tool(monkeypatch):
    """Repeat trap removed — exact same tool+args can run again."""
    calls = {"n": 0}

    def _search(*, query: str = "", limit: int = 3) -> str:
        calls["n"] += 1
        return f"hit:{query}"

    monkeypatch.setattr(
        "src.tools.lesson_tools.search_lessons_impl", _search, raising=True
    )
    app = _prep_graph(
        monkeypatch,
        [
            _tool_ai("a", call_id="c1"),
            _tool_ai("a", call_id="c2"),
            AIMessage(content="Stop repeating — here is the summary."),
        ],
    )
    out = _invoke(app)
    msgs = out.get("messages") or []
    assert calls["n"] == 2
    assert not any(
        isinstance(m, ToolMessage) and "Repeated the same call" in str(m.content)
        for m in msgs
    )
    last = msgs[-1]
    assert isinstance(last, AIMessage)
    assert not getattr(last, "tool_calls", None)
    assert "summary" in str(last.content)


@pytest.mark.integration
def test_cap_stops_endless_loop(monkeypatch):
    monkeypatch.setattr(settings, "max_tool_rounds_per_turn", 4, raising=False)
    calls = {"n": 0}

    def _search(*, query: str = "", limit: int = 3) -> str:
        calls["n"] += 1
        return f"hit:{query}"

    monkeypatch.setattr(
        "src.tools.lesson_tools.search_lessons_impl", _search, raising=True
    )
    script = [_tool_ai(f"q{i}", call_id=f"c{i}") for i in range(8)]
    script.append(AIMessage(content="Stop — limit, here is prose."))
    app = _prep_graph(monkeypatch, script)
    out = _invoke(app)
    msgs = out.get("messages") or []
    assert any(
        isinstance(m, ToolMessage) and "tool_round_limit_reached" in str(m.content)
        for m in msgs
    )
    last = msgs[-1]
    assert isinstance(last, AIMessage)
    assert not getattr(last, "tool_calls", None)
    assert "prose" in str(last.content).casefold() or "Stop" in str(last.content)
    assert 1 <= calls["n"] < 8
