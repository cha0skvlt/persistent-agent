"""Tests for reply locale hints — English-only agent."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from src.agent_reply import extract_agent_reply_text
from src.text_locale import turn_uses_cyrillic


def test_turn_uses_cyrillic():
    # Built from escapes so the source file stays Latin-only.
    cyrillic = "\u0441\u043a\u0430\u0436\u0438 \u0438\u0437 \u043f\u0430\u043c\u044f\u0442\u0438"
    assert turn_uses_cyrillic(cyrillic)
    assert not turn_uses_cyrillic("tell me from memory")


def test_extract_agent_reply_preserves_model_text():
    msgs = [
        HumanMessage(content="who are you"),
        AIMessage(content="You are right. 精神上的转化。"),
    ]
    out = extract_agent_reply_text(msgs)
    assert "精神" in out
    assert "You are right" in out
