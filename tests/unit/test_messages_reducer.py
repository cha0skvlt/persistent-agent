from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage

from src.config import settings
from src.state.agent_state import (
    _is_archived_summary,
    _is_tool_call_pair_start,
    _messages_reducer,
    checkpoint_overflow_updates,
    prune_messages_for_llm,
    summarize_session_trim_updates,
)


def _tool_turn(n: int) -> list:
    tid = f"call-{n}"
    return [
        HumanMessage(content=f"user-{n}"),
        AIMessage(
            content="",
            tool_calls=[
                {"name": "save_memory", "id": tid, "args": {"content": f"f{n}"}}
            ],
        ),
        ToolMessage(
            content="Fact saved successfully",
            tool_call_id=tid,
            name="save_memory",
        ),
        AIMessage(content=f"agent-{n}"),
    ]


def test_prune_30_tool_turns_caps_and_collapses_old_pairs(monkeypatch):
    monkeypatch.setattr(settings, "checkpoint_max_messages", 12, raising=False)
    monkeypatch.setattr(settings, "tool_collapse_keep_recent_pairs", 2, raising=False)

    merged: list = []
    for n in range(30):
        merged.extend(_tool_turn(n))

    out = prune_messages_for_llm(merged)
    assert len(out) <= 12
    assert any(_is_archived_summary(m) for m in out)

    recent = out[-8:]
    full_pairs = sum(
        1
        for i, m in enumerate(recent)
        if _is_tool_call_pair_start(m)
        and i + 1 < len(recent)
        and isinstance(recent[i + 1], ToolMessage)
    )
    assert full_pairs >= 1

    for i, m in enumerate(out):
        if _is_tool_call_pair_start(m):
            assert i + 1 < len(out)
            assert isinstance(out[i + 1], ToolMessage)


def test_prune_anchors_human_when_tool_rounds_exceed_cap(monkeypatch):
    """Regression: Qwen template raises 'No user query found' if the kept window
    has no user turn. Many tool rounds must not drop the anchoring HumanMessage."""
    monkeypatch.setattr(settings, "checkpoint_max_messages", 12, raising=False)
    # High keep-pairs → no collapse; the [-12:] cap alone would drop the head human.
    monkeypatch.setattr(settings, "tool_collapse_keep_recent_pairs", 20, raising=False)

    msgs: list = [HumanMessage(content="reflect freely", id="h0")]
    for n in range(8):
        tid = f"c{n}"
        msgs.append(
            AIMessage(
                content="", tool_calls=[{"name": "search_diary", "id": tid, "args": {}}]
            )
        )
        msgs.append(ToolMessage(content="[]", tool_call_id=tid, name="search_diary"))

    out = prune_messages_for_llm(msgs)
    assert len(out) <= 12
    assert any(
        isinstance(m, HumanMessage) for m in out
    ), "kept window must contain a user query for the Qwen chat template"


def test_prune_idempotent_on_archived_summaries(monkeypatch):
    monkeypatch.setattr(settings, "checkpoint_max_messages", 20, raising=False)
    monkeypatch.setattr(settings, "tool_collapse_keep_recent_pairs", 0, raising=False)

    merged = _tool_turn(1) + _tool_turn(2)
    once = prune_messages_for_llm(merged)
    twice = prune_messages_for_llm(once)
    assert len(twice) == len(once)
    assert sum(_is_archived_summary(m) for m in twice) == sum(
        _is_archived_summary(m) for m in once
    )


def test_messages_reducer_applies_prune(monkeypatch):
    monkeypatch.setattr(settings, "checkpoint_max_messages", 12, raising=False)
    monkeypatch.setattr(settings, "tool_collapse_keep_recent_pairs", 2, raising=False)

    left = []
    right = []
    for n in range(30):
        right.extend(_tool_turn(n))
    out = list(_messages_reducer(left, right))
    assert len(out) <= 12


def test_collapse_multiple_tool_results_in_one_run(monkeypatch):
    monkeypatch.setattr(settings, "checkpoint_max_messages", 30, raising=False)
    monkeypatch.setattr(settings, "tool_collapse_keep_recent_pairs", 0, raising=False)

    merged = [
        HumanMessage(content="go"),
        AIMessage(
            content="",
            tool_calls=[
                {"name": "search_memory", "id": "a", "args": {"query": "x"}},
                {"name": "search_graph", "id": "b", "args": {"query": "y"}},
            ],
        ),
        ToolMessage(content="[]", tool_call_id="a", name="search_memory"),
        ToolMessage(content="[]", tool_call_id="b", name="search_graph"),
        AIMessage(content="done"),
    ]
    out = prune_messages_for_llm(merged)
    archived = [m for m in out if _is_archived_summary(m)]
    assert len(archived) == 1
    assert "search_memory" in archived[0].content
    assert "search_graph" in archived[0].content
    assert not any(isinstance(m, ToolMessage) for m in out)


def test_summarize_tool_pair_format():
    call = AIMessage(
        content="",
        tool_calls=[{"name": "save_memory", "id": "x", "args": {}}],
    )
    result = ToolMessage(
        content="Fact saved successfully", tool_call_id="x", name="save_memory"
    )
    from src.state.agent_state import _summarize_tool_pair

    summary = _summarize_tool_pair(call, result)
    assert _is_archived_summary(summary)
    assert "save_memory" in summary.content
    assert "Fact saved" in summary.content


def test_messages_reducer_honors_remove_message(monkeypatch):
    monkeypatch.setattr(settings, "checkpoint_max_messages", 50, raising=False)
    left = [
        HumanMessage(content="drop", id="a"),
        HumanMessage(content="keep", id="b"),
    ]
    right = [RemoveMessage(id="a")]
    out = list(_messages_reducer(left, right))
    assert len(out) == 1
    assert out[0].content == "keep"


def test_checkpoint_overflow_updates(monkeypatch):
    monkeypatch.setattr(settings, "checkpoint_max_messages", 3, raising=False)
    monkeypatch.setattr(settings, "tool_collapse_keep_recent_pairs", 0, raising=False)
    msgs = [HumanMessage(content=f"m{i}", id=f"id{i}") for i in range(8)]
    ops = checkpoint_overflow_updates(msgs)
    assert ops
    out = list(_messages_reducer(msgs, ops))
    assert len(out) <= 3


def test_summarize_session_trim_keeps_recent_exchange(monkeypatch):
    monkeypatch.setattr(settings, "checkpoint_max_messages", 20, raising=False)
    monkeypatch.setattr(settings, "tool_collapse_keep_recent_pairs", 0, raising=False)
    msgs = [
        HumanMessage(content="pre", id="pre"),
        HumanMessage(content="s0", id="s0"),
        HumanMessage(content="s1", id="s1"),
        HumanMessage(content="s2", id="s2"),
        HumanMessage(content="s3", id="s3"),
    ]
    ops = summarize_session_trim_updates(
        msgs, offset=1, session_msgs=msgs[1:], max_ctx=12, keep_recent=2
    )
    assert ops
    out = list(_messages_reducer(msgs, ops))
    assert len(out) == 3
    assert out[0].content == "pre"
    assert out[1].content == "s2"
    assert out[2].content == "s3"


def test_messages_reducer_empty_right(monkeypatch):
    monkeypatch.setattr(settings, "checkpoint_max_messages", 2, raising=False)
    left = [
        HumanMessage(content="a"),
        HumanMessage(content="b"),
        HumanMessage(content="c"),
    ]
    out = list(_messages_reducer(left, []))
    assert len(out) <= 2


def test_checkpoint_overflow_disabled_when_max_zero(monkeypatch):
    monkeypatch.setattr(settings, "checkpoint_max_messages", 0, raising=False)
    msgs = [HumanMessage(content="x", id="a")]
    assert checkpoint_overflow_updates(msgs) is None


def test_reconcile_skips_phantom_remove_ids(monkeypatch):
    """Regression: do not RemoveMessage ids invented for id-less collapsed messages."""
    monkeypatch.setattr(settings, "checkpoint_max_messages", 2, raising=False)
    monkeypatch.setattr(settings, "tool_collapse_keep_recent_pairs", 0, raising=False)
    msgs = [
        HumanMessage(content="drop-me", id="drop-id"),
        AIMessage(content="[archived: save_memory → 'ok']"),
        HumanMessage(content="keep-me", id="keep-id"),
        HumanMessage(content="tail", id="tail-id"),
    ]
    ops = checkpoint_overflow_updates(msgs)
    assert ops
    remove_ids = {m.id for m in ops if isinstance(m, RemoveMessage)}
    assert "drop-id" in remove_ids
    assert all(rid in {"drop-id", "keep-id", "tail-id"} for rid in remove_ids)
    out = list(_messages_reducer(msgs, ops))
    assert len(out) <= 2
    assert not any(getattr(m, "id", None) == "drop-id" for m in out)


def test_messages_reducer_ignores_stale_remove_message(monkeypatch):
    monkeypatch.setattr(settings, "checkpoint_max_messages", 50, raising=False)
    left = [HumanMessage(content="keep", id="b")]
    right = [RemoveMessage(id="already-gone")]
    out = list(_messages_reducer(left, right))
    assert len(out) == 1
    assert out[0].content == "keep"
