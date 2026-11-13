from __future__ import annotations

from langchain_core.messages import HumanMessage

from src.config import settings
from src.graph.time_node import update_current_time
from src.state.agent_state import _messages_reducer


def test_update_current_time_trims_stale_checkpoint(monkeypatch):
    monkeypatch.setattr(settings, "checkpoint_max_messages", 4, raising=False)
    monkeypatch.setattr(settings, "tool_collapse_keep_recent_pairs", 0, raising=False)
    msgs = [HumanMessage(content=f"m{i}", id=f"id{i}") for i in range(10)]
    state = {
        "messages": msgs,
        "session_uuid_anchor": "same",
        "session_uuid": "same",
        "session_message_offset": 0,
        "turn_count": 1,
    }
    out = update_current_time(state)  # type: ignore[arg-type]
    assert "messages" in out
    trimmed = list(_messages_reducer(msgs, out["messages"]))
    assert len(trimmed) <= 4


def test_update_current_time_trims_on_session_uuid_change(monkeypatch):
    monkeypatch.setattr(settings, "checkpoint_max_messages", 4, raising=False)
    monkeypatch.setattr(settings, "tool_collapse_keep_recent_pairs", 0, raising=False)
    msgs = [HumanMessage(content=f"m{i}", id=f"id{i}") for i in range(10)]
    state = {
        "messages": msgs,
        "session_uuid_anchor": "old",
        "session_uuid": "new",
        "session_message_offset": 0,
    }
    out = update_current_time(state)  # type: ignore[arg-type]
    assert "messages" in out


def test_update_current_time_trims_without_session_uuid(monkeypatch):
    monkeypatch.setattr(settings, "checkpoint_max_messages", 4, raising=False)
    monkeypatch.setattr(settings, "tool_collapse_keep_recent_pairs", 0, raising=False)
    msgs = [HumanMessage(content=f"m{i}", id=f"id{i}") for i in range(10)]
    state = {"messages": msgs}
    out = update_current_time(state)  # type: ignore[arg-type]
    assert "messages" in out


def test_update_current_time_resets_offset_when_session_uuid_changes():
    state = {
        "messages": [HumanMessage(content="old")] * 5 + [HumanMessage(content="new")],
        "session_message_offset": 0,
        "session_uuid_anchor": "session-a",
        "session_uuid": "session-b",
    }
    out = update_current_time(state)  # type: ignore[arg-type]
    assert out["session_uuid_anchor"] == "session-b"
    assert out["session_uuid"] == "session-b"
    assert out["session_message_offset"] == len(state["messages"])
    assert out["turn_count"] == 1


def test_update_current_time_preserves_offset_within_same_session_uuid():
    state = {
        "messages": [HumanMessage(content="x")] * 4,
        "session_message_offset": 2,
        "session_uuid_anchor": "same",
        "session_uuid": "same",
        "turn_count": 9,
    }
    out = update_current_time(state)  # type: ignore[arg-type]
    assert out["session_message_offset"] == 2
    assert out["turn_count"] == 10


def test_update_current_time_legacy_no_session_uuid():
    state = {"messages": [HumanMessage(content="a"), HumanMessage(content="b")]}
    out = update_current_time(state)  # type: ignore[arg-type]
    assert out["session_message_offset"] == 2
    assert out["turn_count"] == 1


def test_update_current_time_bumps_turn_count_again():
    state = {
        "messages": [HumanMessage(content="a")],
        "session_message_offset": 0,
        "turn_count": 4,
    }
    out = update_current_time(state)  # type: ignore[arg-type]
    assert out["turn_count"] == 5
