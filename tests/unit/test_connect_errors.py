"""Ollama / HTTP connection error helpers."""

from __future__ import annotations

import httpx

from src.config import settings
from src.connect_errors import (
    is_transient_connection_error,
    ollama_unreachable_user_message,
)


def test_connect_error_detected():
    assert is_transient_connection_error(
        httpx.ConnectError("All connection attempts failed")
    )


def test_connection_error_detected():
    assert is_transient_connection_error(ConnectionError("Failed to connect to Ollama"))


def test_ollama_message_telegram():
    msg = ollama_unreachable_user_message(settings=settings, channel="telegram")
    assert "Ollama" in msg
    assert "unreachable" in msg
    assert settings.ollama_base_url in msg


def test_ollama_message_no_traceback():
    from src.agent_errors import format_agent_turn_error

    text = format_agent_turn_error(
        httpx.ConnectError("All connection attempts failed"),
        channel="telegram",
    )
    assert "unreachable" in text
    assert "```" not in text


def test_format_agent_turn_error_telegram_includes_traceback():
    from src.agent_errors import format_agent_turn_error

    try:
        raise ValueError("boom")
    except ValueError as exc:
        text = format_agent_turn_error(exc, channel="telegram")
    assert "Turn error" in text
    assert "ValueError" in text
    assert "boom" in text
    assert "Traceback" in text or "ValueError" in text


def test_format_empty_turn_fallback_is_operator_notice():
    from langchain_core.messages import HumanMessage, ToolMessage

    from src.agent_errors import format_empty_turn_fallback

    msgs = [
        HumanMessage(content="x"),
        ToolMessage(content="not_found", tool_call_id="c1", name="forget_diary"),
    ]
    out = format_empty_turn_fallback(msgs, channel="telegram")
    assert "Operator:" in out
    assert "No visible model output" in out
    assert "Turn error" not in out
    assert "forget_diary" in out


def test_looks_like_operator_error_ollama_template():
    from src.agent_errors import looks_like_operator_error

    blob = (
        '{"error":{"code":400,"message":"Unable to generate parser for this template. '
        'No user query found in messages.","type":"invalid_request_error"}}'
    )
    assert looks_like_operator_error(blob) is True
    assert looks_like_operator_error("**Turn error** `ResponseError`") is True
    assert looks_like_operator_error(
        "Operator: No visible model output — the turn ended without text."
    )
    assert looks_like_operator_error("I was thinking about my own nature") is False
