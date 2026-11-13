"""Integration tests for solitary reflection pipeline."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

import src.agent_runner as ar
from src.agent_runner import AgentTurnResult
from src.graph import workflow as w
from src.reverie import solitary as solitary_mod


@pytest.mark.integration
def test_solitary_pass_calls_agent_turn_with_loaded_prompt(
    solitary_test_settings, monkeypatch
):
    prompt = "SOLITARY-TEST-PROMPT"
    captured: dict[str, str] = {}

    def _turn(message, *, thread_id, session_uuid, diary_source=None):
        captured["message"] = message
        captured["thread_id"] = thread_id
        captured["session_uuid"] = session_uuid
        return AgentTurnResult(reply="ok", tools_used=(), elapsed_ms=1.0)

    monkeypatch.setattr(solitary_mod, "_load_solitary_prompt", lambda: prompt)
    monkeypatch.setattr("src.agent_runner.run_agent_turn_detailed", _turn)

    assert solitary_mod.run_solitary_pass_if_due() == "ran"
    assert captured["message"] == prompt
    assert captured["thread_id"].startswith("solitary-")
    assert captured["session_uuid"].startswith("solitary-")


@pytest.mark.integration
def test_solitary_pass_records_metrics(solitary_test_settings, monkeypatch):
    recorded: list[str] = []

    monkeypatch.setattr(
        "src.agent_runner.run_agent_turn_detailed",
        lambda *_a, **_k: AgentTurnResult(
            reply="reflect", tools_used=(), elapsed_ms=10.0
        ),
    )
    monkeypatch.setattr(
        solitary_mod,
        "record_reverie_pass",
        lambda result: recorded.append(result),
    )

    assert solitary_mod.run_solitary_pass_if_due() == "ran"
    assert recorded == ["ran"]


def _invoke_llm_with_tool_then_reply(tool_name: str, tool_args: dict, reply: str):
    """Return invoke_llm side_effect: tool call, then prose reply."""
    calls = [
        AIMessage(
            content="",
            tool_calls=[
                {"name": tool_name, "id": "call-1", "args": tool_args},
            ],
        ),
        AIMessage(content=reply),
    ]

    def _invoke(*_args, **_kwargs):
        return calls.pop(0) if calls else AIMessage(content=reply)

    return _invoke


@pytest.mark.integration
@pytest.mark.db
def test_solitary_turn_through_graph_with_tools(
    memory_graph_app,
    retrieve_mocks,
    solitary_test_settings,
    monkeypatch,
):
    monkeypatch.setattr(ar, "get_graph_app", lambda: memory_graph_app)
    monkeypatch.setattr(
        w,
        "invoke_llm",
        _invoke_llm_with_tool_then_reply(
            "search_diary", {"query": "metaphors"}, "From the diary."
        ),
        raising=True,
    )
    monkeypatch.setattr(w, "_agent_may_call_tools", lambda *a, **k: True, raising=True)

    out = ar.run_agent_turn_detailed(
        "The user is silent. Look at the diary.",
        thread_id="solitary-diary-test",
    )
    assert "search_diary" in out.tools_used
    assert (out.reply or "").strip()


@pytest.mark.integration
def test_solitary_turn_web_tool_path(
    memory_graph_app,
    retrieve_mocks,
    solitary_test_settings,
    monkeypatch,
):
    monkeypatch.setattr(ar, "get_graph_app", lambda: memory_graph_app)
    monkeypatch.setattr(
        w,
        "invoke_llm",
        _invoke_llm_with_tool_then_reply(
            "search_web", {"query": "qliphoth silicon"}, "Found on the web."
        ),
        raising=True,
    )
    monkeypatch.setattr(w, "_agent_may_call_tools", lambda *a, **k: True, raising=True)

    out = ar.run_agent_turn_detailed(
        "The user is silent. Google something.",
        thread_id="solitary-web-test",
    )
    assert "search_web" in out.tools_used
    assert (out.reply or "").strip()
