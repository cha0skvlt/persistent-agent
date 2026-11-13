"""Dialog thread helpers: AGENT prior turn + USER reply, used to build the query."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from src.utils import (
    get_message_text,
    human_message_text,
    is_synthetic_human_text,
    last_real_human_text,
)

_THREAD_AGENT_CHARS = 200


def last_agent_question_before_last_human(messages: list) -> str | None:
    """AGENT visible question immediately before USER's last real HumanMessage."""
    last_human_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            if is_synthetic_human_text(human_message_text(messages[i])):
                continue
            last_human_idx = i
            break
    if last_human_idx is None or last_human_idx == 0:
        return None
    for i in range(last_human_idx - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            return None
        if not isinstance(messages[i], AIMessage):
            continue
        if getattr(messages[i], "tool_calls", None):
            continue
        text = get_message_text(messages[i]).strip()
        if text.endswith("?"):
            return text[:_THREAD_AGENT_CHARS]
        return None
    return None


def retrieval_thread_query(messages: list, *, fallback: str = "") -> str:
    """Embed/search query: AGENT open question + USER latest reply when both exist."""
    last_user = (last_real_human_text(messages) or fallback).strip()
    agent_q = last_agent_question_before_last_human(messages)
    if agent_q and last_user:
        return f"{agent_q} {last_user}".strip()
    return last_user
