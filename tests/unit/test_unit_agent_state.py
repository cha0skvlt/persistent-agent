"""Unit tests for agent_state reducers and message helpers."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.config import settings
from src.state import agent_state as ast


@pytest.mark.unit
def test_reducers_merge_and_replace():
    assert ast.prune_messages_for_llm([]) == []


@pytest.mark.unit
def test_message_text_and_tool_preview():
    class BlockMsg:
        content = ["plain", {"text": "t"}]

    assert ast._message_text(BlockMsg()) == "plaint"
    assert ast._message_text(type("X", (), {"content": 99})()) == "99"

    think_body = "<think>debug</think>\n" "visible reply"
    assert ast._message_text(AIMessage(content=think_body)) == "visible reply"

    long = "x" * 100
    prev = ast._tool_result_preview(ToolMessage(content=long, tool_call_id="1"))
    assert prev.endswith("…") and len(prev) < len(long)


@pytest.mark.unit
def test_summarize_tool_run_branches():
    call = AIMessage.model_construct(
        content="",
        tool_calls=[
            {"name": "n1", "id": "i1", "args": {}},
            object(),
            {"name": "n2", "id": "i2", "args": {}},
        ],
    )
    t1 = ToolMessage(content="ok1", tool_call_id="i1", name="n1")
    t2 = ToolMessage(content="ok2", tool_call_id="i2", name="n2")
    msg = ast._summarize_tool_run(call, [t1, t2])
    assert "n1" in msg.content and "n2" in msg.content

    call2 = AIMessage(content="", tool_calls=[])
    out2 = ast._summarize_tool_run(call2, [t1])
    assert "n1" in out2.content or "ok1" in out2.content

    orphan = AIMessage.model_construct(
        content="",
        tool_calls=[{"name": "x", "id": "no", "args": {}}],
    )
    out3 = ast._summarize_tool_run(
        orphan,
        [ToolMessage(content="r", tool_call_id="other", name="tn")],
    )
    assert "tn" in out3.content


@pytest.mark.unit
def test_message_text_list_and_non_str_content():
    class BlockMsg:
        content = [
            "<think>t</think>\n",
            {"text": "answer"},
        ]

    assert ast._message_text(BlockMsg()) == "answer"
    assert ast._message_text(type("X", (), {"content": 99})()) == "99"


@pytest.mark.unit
def test_is_archived_and_tool_pair_prune(monkeypatch):
    monkeypatch.setattr(settings, "checkpoint_max_messages", 100)
    monkeypatch.setattr(settings, "tool_collapse_keep_recent_pairs", 1)
    call = AIMessage(
        content="",
        tool_calls=[{"name": "old", "id": "1", "args": {}}],
    )
    old_tool = ToolMessage(content="otr", tool_call_id="1")
    call2 = AIMessage(
        content="",
        tool_calls=[{"name": "new", "id": "2", "args": {}}],
    )
    new_tool = ToolMessage(content="ntr", tool_call_id="2")
    archived = AIMessage(content=ast._ARCHIVED_PREFIX + " shadow")
    seq = [archived, call, old_tool, call2, new_tool]
    out = ast.prune_messages_for_llm(seq)
    assert len(out) <= 100
    assert any(ast._is_archived_summary(m) for m in out)


@pytest.mark.unit
def test_summarize_tool_pair_direct():
    call = AIMessage(
        content="",
        tool_calls=[{"name": "probe", "id": "z", "args": {}}],
    )
    res = ToolMessage(content="result text", tool_call_id="z", name="probe")
    msg = ast._summarize_tool_pair(call, res)
    assert ast._ARCHIVED_PREFIX in msg.content


@pytest.mark.unit
def test_human_not_tool_call_pair_start():
    assert ast._is_tool_call_pair_start(HumanMessage(content="x")) is False


@pytest.mark.unit
def test_summarize_session_trim_keep_recent_zero():
    msgs = [HumanMessage(content="a"), AIMessage(content="b")]
    ops = ast.summarize_session_trim_updates(
        msgs,
        offset=0,
        session_msgs=msgs,
        max_ctx=10,
        keep_recent=0,
    )
    assert ops is None or isinstance(ops, list)


def test_messages_reducer_appends(monkeypatch):
    monkeypatch.setattr(settings, "checkpoint_max_messages", 50)
    monkeypatch.setattr(settings, "tool_collapse_keep_recent_pairs", 2)
    left = [HumanMessage(content="a")]
    right = [HumanMessage(content="b")]
    out = ast._messages_reducer(left, right)
    assert len(out) >= 2
