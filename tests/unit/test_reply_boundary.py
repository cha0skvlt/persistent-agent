"""Visible reply boundary: planning must never be promoted to speech."""

from __future__ import annotations

import pytest

from src.reply.visible import (
    _is_speech_paragraph,
    promote_speech_from_thinking,
    strip_leaked_tool_markup,
)


@pytest.mark.unit
def test_bare_tool_calls_are_stripped():
    text = 'search_memory(q="self identity agent")\nsearch_diary(q="identity self")'
    assert strip_leaked_tool_markup(text) == ""


@pytest.mark.unit
def test_bare_tool_result_echoes_are_stripped():
    text = (
        "`search_memory` -> `[]`\n"
        "`search_diary` -> `[]`\n"
        "`search_graph` -> `[]`\n\n"
        "I am a persistent agent with no name."
    )
    out = strip_leaked_tool_markup(text)
    assert "search_memory" not in out
    assert "I am a persistent agent" in out


@pytest.mark.unit
def test_tool_and_args_labels_are_stripped():
    text = (
        "Tool: `save_memory`\n"
        'Args: `text="USER: prefers dark mode"`\n\n'
        'Reply: "I have recorded that."'
    )
    out = strip_leaked_tool_markup(text)
    assert "save_memory" not in out
    assert "Args:" not in out
    assert "I have recorded that." in out


@pytest.mark.unit
def test_tool_output_label_is_stripped():
    text = "tool_output:\n[]"
    assert strip_leaked_tool_markup(text) == ""


@pytest.mark.unit
def test_bare_tool_names_on_their_own_lines_are_stripped():
    text = "`search_memory`\n`search_diary`\n`search_lessons`"
    assert strip_leaked_tool_markup(text) == ""


@pytest.mark.unit
def test_prose_mentioning_a_tool_name_survives():
    text = "I can search_memory when you ask me to recall something."
    assert "search_memory" in strip_leaked_tool_markup(text)


@pytest.mark.unit
def test_bare_call_is_recovered_as_tool_call():
    from langchain_core.messages import AIMessage

    from src.graph.htc_recovery import try_materialize_htc_tool_call

    msg = AIMessage(content='search_memory(query="name OR identity")')
    out = try_materialize_htc_tool_call(
        msg, allowed_tool_names=frozenset({"search_memory"})
    )
    assert out is not None
    assert out.tool_calls[0]["name"] == "search_memory"
    assert out.tool_calls[0]["args"]["query"] == "name OR identity"


@pytest.mark.unit
def test_positional_bare_call_is_recovered():
    from langchain_core.messages import AIMessage

    from src.graph.htc_recovery import try_materialize_htc_tool_call

    msg = AIMessage(
        content='save_memory(content, "USER: the operator prefers dark mode.")'
    )
    out = try_materialize_htc_tool_call(
        msg, allowed_tool_names=frozenset({"save_memory"})
    )
    assert out is not None
    assert (
        out.tool_calls[0]["args"]["content"] == "USER: the operator prefers dark mode."
    )


@pytest.mark.unit
def test_save_memory_accepts_text_alias():
    from src.graph.htc_recovery import validate_recovered_tool_args

    assert validate_recovered_tool_args("save_memory", {"text": "hello"}) == {
        "content": "hello"
    }


@pytest.mark.unit
def test_unknown_bare_call_is_not_recovered():
    from langchain_core.messages import AIMessage

    from src.graph.htc_recovery import try_materialize_htc_tool_call

    msg = AIMessage(content='rm_rf(path="/tmp")')
    assert (
        try_materialize_htc_tool_call(
            msg, allowed_tool_names=frozenset({"search_memory"})
        )
        is None
    )


@pytest.mark.unit
def test_synthesize_from_empty_tool_result():
    from langchain_core.messages import ToolMessage

    from src.graph.empty_reply_gate import synthesize_from_tool_result

    msgs = [
        ToolMessage(content="No triples matched", tool_call_id="c1", name="query_graph")
    ]
    assert synthesize_from_tool_result(msgs) is not None


@pytest.mark.unit
def test_synthesize_ignores_useful_tool_result():
    from langchain_core.messages import ToolMessage

    from src.graph.empty_reply_gate import synthesize_from_tool_result

    msgs = [ToolMessage(content="saved id=abc", tool_call_id="c1", name="save_memory")]
    assert synthesize_from_tool_result(msgs) is None


@pytest.mark.unit
def test_diary_save_tool_is_registered():
    from src.tools.diary_tools import diary_save

    assert diary_save.name == "diary_save"


@pytest.mark.unit
def test_planning_paragraph_is_not_speech():
    plan = (
        "The user is asking me to describe myself. I need to check my memory "
        "and diary to understand my identity and state. I will search memory "
        "for any self-description."
    )
    assert _is_speech_paragraph(plan) is False


@pytest.mark.unit
def test_real_reply_is_speech():
    reply = "I am a persistent agent with no name, and my memory lives in a database."
    assert _is_speech_paragraph(reply) is True


@pytest.mark.unit
def test_planning_is_not_promoted_from_thinking():
    think = (
        "The user is asking me to describe myself.\n\n"
        "I need to check my memory and diary.\n\n"
        "I will search memory for any self-description.\n\n"
        "Let's do that."
    )
    assert promote_speech_from_thinking(think, "") is None


@pytest.mark.unit
def test_real_answer_in_thinking_is_promoted():
    think = (
        "The user asks what I am.\n\n"
        "I should answer plainly.\n\n"
        "I am a persistent agent. My memory lives in a PostgreSQL database."
    )
    out = promote_speech_from_thinking(think, "")
    assert out is not None
    assert "persistent agent" in out[1]
