"""LLM_SUPPORTS_TOOLS=0 — no bind_tools; HTC recovery still executes tools."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.graph.route_turn_node import route_turn_node


def test_get_agent_llm_with_tools_skips_bind_when_unsupported(monkeypatch):
    import src.llm_factory as lf

    monkeypatch.setattr(lf.get_settings(), "llm_supports_tools", False, raising=False)
    lf.clear_llm_cache()

    tools_llm = lf.get_agent_tools_llm()
    with_tools = lf.get_agent_llm_with_tools([object()])
    assert with_tools is tools_llm
    assert with_tools is not lf.get_agent_llm()


def test_get_agent_llm_with_tools_binds_when_supported(monkeypatch):
    import src.llm_factory as lf

    monkeypatch.setattr(lf.get_settings(), "llm_supports_tools", True, raising=False)
    lf.clear_llm_cache()

    calls: list[list] = []

    class _FakeLLM:
        def bind_tools(self, tools):
            calls.append(list(tools))
            return ("bound", tools)

    monkeypatch.setattr(lf, "get_agent_tools_llm", lambda: _FakeLLM())
    out = lf.get_agent_llm_with_tools(["t1"])
    assert calls == [["t1"]]
    assert out == ("bound", ["t1"])


def test_tools_bypass_htc_json_recovers_and_tool_runs(monkeypatch):
    """LLM_SUPPORTS_TOOLS=0: unbound LLM emits HTC JSON → recovery → ToolNode."""
    import src.graph.workflow as w
    import src.llm_factory as lf

    monkeypatch.setattr(w.settings, "llm_supports_tools", False, raising=False)
    monkeypatch.setattr(lf.get_settings(), "llm_supports_tools", False, raising=False)
    lf.clear_llm_cache()
    monkeypatch.setattr(w.settings, "reflection", False, raising=False)

    htc_body = (
        '{"name": "search_web", "parameters": {"query": "LangGraph latest version"}}'
    )

    def fake_invoke(_inv, _msgs, **kw):
        return AIMessage(content=htc_body)

    monkeypatch.setattr(w, "invoke_llm", fake_invoke)
    monkeypatch.setattr(w, "_agent_may_call_tools", lambda *a, **k: True)

    # Prove factory path returns unbound agent LLM (no bind_tools).
    assert (
        w.llm_factory.get_agent_llm_with_tools(w.tools)
        is w.llm_factory.get_agent_tools_llm()
    )

    messages = [HumanMessage(content="google the latest LangGraph version")]
    route = route_turn_node(
        {
            "messages": messages,
            "current_time": "2026-07-02",
        }
    )
    state = {
        "messages": messages,
        "current_time": "2026-07-02",
        "retrieved_memories": [],
        "retrieved_self_reflections": [],
        "turn_count": 1,
        "tool_rounds": 0,
        **route,
    }
    out = w.agent_node(state)
    ai = out["messages"][-1]
    assert isinstance(ai, AIMessage)
    assert ai.tool_calls
    assert ai.tool_calls[0]["name"] == "search_web"
    assert str(ai.tool_calls[0]["id"]).startswith("htc_")

    executed: list[str] = []

    def fake_tool_invoke(tool_node, tool_state, config):
        for m in tool_state.get("messages") or []:
            for tc in getattr(m, "tool_calls", None) or []:
                if isinstance(tc, dict) and tc.get("name"):
                    executed.append(str(tc["name"]))
        return {
            "messages": [
                ToolMessage(
                    content="ok: search_web ran",
                    tool_call_id=ai.tool_calls[0]["id"],
                    name="search_web",
                )
            ]
        }

    monkeypatch.setattr(w, "invoke_tool_node_logged", fake_tool_invoke)
    tool_state = {
        **state,
        "messages": list(state["messages"]) + [ai],
        "tool_rounds": 0,
    }
    tool_out = w.tools_node(tool_state, {})
    assert executed == ["search_web"]
    assert any(
        isinstance(m, ToolMessage) and "search_web ran" in str(m.content)
        for m in tool_out["messages"]
    )
