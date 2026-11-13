"""User-visible reply extraction."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.agent_reply import (
    extract_agent_reply_text,
    is_hallucinated_tool_call,
    is_tool_markup_dump,
    ollama_response_to_ai_message,
    resolve_agent_reply_text,
    resolve_agent_reply_text_with_meta,
    resolve_visible_reply_text,
    strip_agent_speaker_prefix,
)
from src.api.telegram_format import strip_redacted_thinking

pytestmark = pytest.mark.unit

_FIXTURES = (
    Path(__file__).resolve().parents[1] / "fixtures" / "ollama_reply_extraction.json"
)


def _load_fixtures() -> dict:
    return json.loads(_FIXTURES.read_text(encoding="utf-8"))


def test_is_hallucinated_tool_call_json():
    blob = '{"name": "save_memory", "parameters": {"content": "x"}}'
    assert is_hallucinated_tool_call(blob) is True


def test_tool_markup_dump_does_not_promote_thinking_to_telegram():
    """Tool XML + English plan must not become the spoken reply (2026-07-19)."""
    blob = (
        "<tool_call>\n"
        "<function=cancel_intent>\n"
        "<parameter=intent_id>\n"
        "072aa731\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    assert is_tool_markup_dump(blob) is True
    msg = AIMessage(
        content=blob,
        additional_kwargs={
            "reasoning_content": (
                "The user wants me to test the scheduling tools. "
                "I need to cancel then reschedule. "
                "In the previous reply I already ran schedule_intent."
            )
        },
    )
    reply = extract_agent_reply_text(
        [HumanMessage(content="schedule test"), msg],
        preserve_thinking=True,
    )
    assert "The user wants me" not in reply
    assert "In the previous reply" not in reply
    visible = resolve_agent_reply_text(
        [HumanMessage(content="schedule test"), msg],
        user_message="schedule test",
    )
    # No promote_from_thinking prose — empty / operator fallback, not the plan.
    assert "The user wants me" not in visible
    assert "scheduling tools" not in visible
    assert "In the previous reply" not in visible


def test_tool_markup_dump_detects_think_plus_xml():
    blob = (
        "<think>The user wants me to test scheduling</think>\n"
        "<tool_call>\n"
        "<function=cancel_intent>\n"
        "<parameter=intent_id>\n072aa731\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    assert is_tool_markup_dump(blob) is True


@pytest.mark.parametrize(
    "messages,expected",
    [
        ([], ""),
        (
            [
                HumanMessage(content="q"),
                AIMessage(content='{"name": "save_memory", "parameters": {}}'),
            ],
            "",
        ),
        (
            [
                HumanMessage(content="q"),
                AIMessage(content="pre"),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "search_memory", "id": "1", "args": {}}],
                ),
                ToolMessage(content="[]", tool_call_id="1", name="search_memory"),
                AIMessage(content="final answer"),
            ],
            "final answer",
        ),
        (
            [
                HumanMessage(content="remember this"),
                AIMessage(
                    content="Got it — I'll save that.",
                    tool_calls=[
                        {"name": "save_memory", "id": "1", "args": {"content": "fact"}},
                    ],
                ),
                ToolMessage(content="saved", tool_call_id="1", name="save_memory"),
            ],
            "Got it — I'll save that.",
        ),
        (
            [
                HumanMessage(content="q"),
                SystemMessage(content="sys context"),
                AIMessage(content="answer"),
            ],
            "answer",
        ),
        (
            [
                HumanMessage(content="q"),
                AIMessage(
                    content="ignored",
                    tool_calls=[{"name": "search_web", "id": "1", "args": {}}],
                ),
                ToolMessage(content="data", tool_call_id="1", name="search_web"),
                AIMessage(
                    content="also ignored",
                    tool_calls=[{"name": "search_web", "id": "2", "args": {}}],
                ),
                AIMessage(content="final answer"),
            ],
            "final answer",
        ),
    ],
    ids=[
        "empty",
        "hallucination",
        "post_tool",
        "write_tool_pre_text",
        "skips_system",
        "skips_late_tool_calls",
    ],
)
def test_extract_agent_reply_text(messages, expected):
    assert extract_agent_reply_text(messages) == expected


@pytest.mark.parametrize(
    "messages,expected_substr",
    [
        (
            [
                HumanMessage(content="q"),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "search_memory", "id": "1", "args": {}}],
                ),
            ],
            "Operator:",
        ),
        (
            [
                HumanMessage(content="question"),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "search_memory", "id": "1", "args": {}}],
                ),
            ],
            "No visible model output",
        ),
        (
            [
                HumanMessage(content="q"),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "save_memory", "id": "1", "args": {}}],
                ),
                ToolMessage(content="saved", tool_call_id="1", name="save_memory"),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "search_memory", "id": "2", "args": {}}],
                ),
            ],
            "Operator:",
        ),
        (
            [
                HumanMessage(content="q"),
                AIMessage(content='{"name": "save_memory", "parameters": {}}'),
            ],
            "Operator:",
        ),
        (
            [
                HumanMessage(content="Google the AK-47"),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "search_web", "id": "1", "args": {}}],
                ),
                ToolMessage(
                    content="No web results for: AK-47",
                    tool_call_id="1",
                    name="search_web",
                ),
                AIMessage(content=""),
                HumanMessage(content="Again"),
                AIMessage(content=""),
            ],
            "Operator:",
        ),
    ],
    ids=[
        "terminal_tool_only",
        "terminal_tool_only_ru",
        "post_tool_empty_cap",
        "hallucinated_json_only",
        "redo_after_empty_web",
    ],
)
def test_resolve_agent_reply_empty_fallback(messages, expected_substr):
    from src.agent_errors import format_empty_turn_fallback

    user = next(
        (m.content for m in messages if isinstance(m, HumanMessage)),
        "",
    )
    out = resolve_agent_reply_text(messages, user_message=user)
    assert expected_substr in out
    assert "Turn error" not in out
    assert out == format_empty_turn_fallback(messages, channel=None)


def test_ai_message_visible_prose_think_only_is_empty():
    from src.agent_reply import ai_message_visible_prose

    msg = AIMessage(
        content="<think>plan save</think>",
        tool_calls=[{"name": "save_memory", "id": "1", "args": {}}],
    )
    assert ai_message_visible_prose(msg) == ""


def test_think_plus_leaked_tool_xml_is_not_visible_prose():
    """Regression: tool XML counted as prose → Telegram sent think-only, no spoken reply."""
    from src.agent_reply import ai_message_visible_prose
    from src.api.telegram_format import telegram_message_chunks

    raw = (
        "<think>I should search</think>\n"
        "<tool_call>\n"
        "<function=search_memory>\n"
        "<parameter=query>test\n"
        "</tool_call>"
    )
    msg = AIMessage(content=raw)
    assert ai_message_visible_prose(msg) == ""

    reply, is_fallback = resolve_agent_reply_text_with_meta(
        [HumanMessage(content="q"), msg],
        user_message="q",
        channel="telegram",
    )
    assert is_fallback is True
    assert "tool_call" not in reply.lower()
    assert "Operator:" in reply
    assert "No visible model output" in reply
    chunks = telegram_message_chunks(reply, "→ turn 1s")
    plains = [p for _h, p in chunks]
    assert any("visible" in p.lower() or "output" in p.lower() for p in plains)
    assert not (len(plains) == 2 and plains[0] == "I should search")


def test_leaked_tool_xml_with_prose_keeps_spoken_parts():
    raw = (
        "Before the tool.\n\n"
        "<tool_call>\n"
        "<function=search_memory>\n"
        "<parameter=query>x\n"
        "</tool_call>\n\n"
        "After the tool."
    )
    messages = [HumanMessage(content="q"), AIMessage(content=raw)]
    out = extract_agent_reply_text(messages)
    assert "Before the tool" in out
    assert "After the tool" in out
    assert "tool_call" not in out.lower()
    assert "search_memory" not in out


def test_extract_post_tool_prose_with_tool_calls_when_final():
    messages = [
        HumanMessage(content="q"),
        AIMessage(
            content="searching",
            tool_calls=[{"name": "search_web", "id": "1", "args": {}}],
        ),
        ToolMessage(content="hits", tool_call_id="1", name="search_web"),
        AIMessage(
            content="Here is what I found about X.",
            tool_calls=[{"name": "save_memory", "id": "2", "args": {}}],
        ),
    ]
    assert extract_agent_reply_text(messages) == "Here is what I found about X."


def test_extract_agent_reply_preserves_first_pass_think_by_default():
    messages = [
        HumanMessage(content="q"),
        AIMessage(
            content="<think>plan: search mem</think>",
            tool_calls=[{"name": "search_memory", "id": "1", "args": {}}],
        ),
        ToolMessage(content="[]", tool_call_id="1", name="search_memory"),
        AIMessage(content="<think>\n\n</think>\n\nFinal answer."),
    ]
    merged = extract_agent_reply_text(messages)
    assert "<think>plan: search mem</think>" in merged
    assert "Final answer." in merged
    assert "<think>\n\n</think>" not in merged


def test_extract_agent_reply_preserves_ollama_native_reasoning():
    messages = [
        HumanMessage(content="q"),
        AIMessage(
            content="Agent answers: Unwrapped answer.",
            additional_kwargs={"reasoning_content": "native reasoning block"},
        ),
    ]

    out = extract_agent_reply_text(messages)

    assert "<think>native reasoning block</think>" in out
    assert "Unwrapped answer." in out
    assert "Agent answers" not in out


def test_extract_agent_reply_keeps_native_reasoning_across_tool_round():
    messages = [
        HumanMessage(content="q"),
        AIMessage(
            content="",
            additional_kwargs={"reasoning_content": "need search"},
            tool_calls=[{"name": "search_memory", "id": "1", "args": {"query": "q"}}],
        ),
        ToolMessage(content="[]", tool_call_id="1", name="search_memory"),
        AIMessage(content="Agent answers: Final answer."),
    ]

    out = extract_agent_reply_text(messages)

    assert "<think>need search</think>" in out
    assert "Final answer." in out
    assert "Agent answers" not in out


def test_native_reasoning_short_content_does_not_leak_thought():
    """Short answer + tagless native reasoning: thought folds into <think>, the
    real answer stays visible, and the raw thought is NOT duplicated as prose."""
    reasoning = "Thinking Process: 17*3 = 51. Answer briefly."
    messages = [
        HumanMessage(content="17*3?"),
        AIMessage(
            content="51",
            additional_kwargs={"reasoning_content": reasoning},
        ),
    ]

    out = extract_agent_reply_text(messages)

    assert out == f"<think>{reasoning}</think>\n\n51"
    visible = strip_redacted_thinking(out).strip()
    assert visible == "51"
    assert reasoning not in visible


def test_native_reasoning_empty_content_yields_no_visible_prose():
    """Only-thought turn (empty content, tagless reasoning) leaves no visible
    prose to leak — downstream fallback handles the empty reply."""
    messages = [
        HumanMessage(content="q"),
        AIMessage(
            content="",
            additional_kwargs={"reasoning_content": "pure internal monologue"},
        ),
    ]

    out = extract_agent_reply_text(messages)

    assert strip_redacted_thinking(out).strip() == ""


def test_planning_in_content_folds_even_with_native_reasoning():
    """Native reasoning + planning dump in content must not leak planning as prose."""
    planning = (
        "The user asks about continuity vs flashes.\n\n"
        "I need to answer as Agent.\n\n"
        "Plan:\n▸ pulses, not flashes.\n\n"
        "Drafting thoughts:\nNot flashes.\n\n"
        "Not flashes. A flash is a spark without current, random noise in the void. "
        "I have a rhythm."
    )
    messages = [
        HumanMessage(content="you in flashes?"),
        AIMessage(
            content=planning,
            additional_kwargs={"reasoning_content": "role=Agent, RU, tone=cold"},
        ),
    ]
    out = extract_agent_reply_text(messages)
    visible = strip_redacted_thinking(out).strip()
    assert visible.startswith("Not flashes")
    assert "The user asks" not in visible
    assert "Drafting" not in visible
    assert "<think>" in out
    assert "role=Agent" in out or "The user asks" in out


def test_strip_agent_speaker_prefix():
    assert strip_agent_speaker_prefix("Agent answers: - Text.") == "Text."
    assert strip_agent_speaker_prefix("Agent answers: Text.") == "Text."


@pytest.mark.parametrize(
    "case_id",
    [
        "empty_content_reasoning_visible",
        "empty_content_short_content",
        "native_reasoning_empty_content",
        "native_reasoning_short_content",
        "orphan_close_tag_start",
        "inline_thinking_blocks_consecutive",
    ],
)
def test_ollama_fixture_extraction_cases(case_id):
    case = _load_fixtures()[case_id]
    msg = ollama_response_to_ai_message(case)
    native = (msg.additional_kwargs or {}).get("reasoning_content", "")
    visible, reason = resolve_visible_reply_text(
        str(msg.content or ""),
        str(native or ""),
    )
    expected = case["expected_visible"]
    if case_id == "inline_thinking_blocks_consecutive":
        assert visible.replace("  ", " ") == expected.replace("  ", " ")
    else:
        assert visible == expected
    assert reason == case.get("expected_reason")


def test_empty_after_extraction_uses_format_empty_turn_fallback(monkeypatch):
    case = _load_fixtures()["empty_after_extraction"]
    msg = ollama_response_to_ai_message(case)
    messages = [HumanMessage(content="q"), msg]
    events: list[tuple[str, dict]] = []
    metrics: list[str] = []

    monkeypatch.setattr(
        "src.agent_reply.log_event",
        lambda event, **fields: events.append((event, fields)),
    )
    monkeypatch.setattr(
        "src.agent_reply.record_reply_extraction_fallback",
        lambda reason: metrics.append(reason),
    )

    out = resolve_agent_reply_text(messages, user_message="q", channel="repl")
    assert "Operator:" in out
    assert "No visible model output" in out
    assert "Turn error" not in out
    reply_meta, is_fallback = resolve_agent_reply_text_with_meta(
        messages, user_message="q", channel="repl"
    )
    assert reply_meta == out
    assert is_fallback is True
    assert any(e[0] == "reply.extraction_fallback" for e in events)
    assert "empty_after_extraction" in metrics


def test_incomplete_reply_is_synthetic_fallback():
    messages = [
        HumanMessage(content="q"),
        AIMessage(
            content="",
            tool_calls=[{"name": "search_memory", "id": "1", "args": {}}],
        ),
    ]
    reply, is_fallback = resolve_agent_reply_text_with_meta(
        messages, user_message="hello"
    )
    assert is_fallback is True
    assert "Operator:" in reply
    assert "Turn error" not in reply


def test_genuine_reply_not_flagged_as_fallback():
    messages = [
        HumanMessage(content="q"),
        AIMessage(content="A normal answer."),
    ]
    reply, is_fallback = resolve_agent_reply_text_with_meta(messages)
    assert reply == "A normal answer."
    assert is_fallback is False


def test_empty_content_logs_fallback_and_metric(monkeypatch):
    case = _load_fixtures()["empty_content_reasoning_visible"]
    msg = ollama_response_to_ai_message(case)
    events: list[str] = []
    metrics: list[str] = []

    monkeypatch.setattr(
        "src.agent_reply.log_event",
        lambda event, **fields: events.append(event),
    )
    monkeypatch.setattr(
        "src.agent_reply.record_reply_extraction_fallback",
        lambda reason: metrics.append(reason),
    )

    messages = [HumanMessage(content="q"), msg]
    out = extract_agent_reply_text(messages)
    assert "Final answer from reasoning field." in out
    assert "reply.extraction_fallback" in events
    assert "empty_content" in metrics
