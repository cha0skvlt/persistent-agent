"""Shared utils (message text, self-referential detection)."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.utils import (
    get_message_text,
    is_synthetic_human_text,
    last_human_text,
    last_real_human_text,
)


@pytest.mark.unit
def test_last_human_text_skips_trailing_ai():
    msgs = [
        HumanMessage(content="first"),
        AIMessage(content="reply"),
        HumanMessage(content="second"),
    ]
    assert last_human_text(msgs) == "second"


@pytest.mark.unit
def test_last_real_human_text_skips_task_reanchor():
    msgs = [
        HumanMessage(content="real question"),
        AIMessage(
            content="", tool_calls=[{"name": "search_web", "id": "1", "args": {}}]
        ),
        ToolMessage(content="results", tool_call_id="1", name="search_web"),
        HumanMessage(content="[Task: summarize the web output]"),
        AIMessage(content="answer"),
    ]
    assert last_human_text(msgs) == "[Task: summarize the web output]"
    assert last_real_human_text(msgs) == "real question"
    assert is_synthetic_human_text("[Task: foo]") is True
    assert is_synthetic_human_text("[Thread: bar]") is True
    assert is_synthetic_human_text("[Deixis: hard]") is True
    assert is_synthetic_human_text("hello") is False


@pytest.mark.unit
def test_get_message_text_transcript_roles():
    assert "USER:" in get_message_text(HumanMessage(content="hi"), style="transcript")
    assert "Tool (" in get_message_text(
        ToolMessage(content="ok", tool_call_id="1", name="search_memory"),
        style="transcript",
    )
    assert get_message_text(HumanMessage(content="hi"), style="plain") == ""
