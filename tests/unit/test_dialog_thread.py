"""Unit tests for dialog thread query helpers (D3)."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from src.graph.dialog_thread import (
    last_agent_question_before_last_human,
    retrieval_thread_query,
)


def test_retrieval_thread_query_concatenates_agent_question_and_user_reply():
    agent_q = "What is the first word that is not a word?"
    user = "The first word was INIT, in Sanskrit OM."
    msgs = [
        HumanMessage(content="You are leading the conversation, agent."),
        AIMessage(content=agent_q),
        HumanMessage(content=user),
    ]
    q = retrieval_thread_query(msgs)
    assert agent_q in q
    assert user in q


def test_retrieval_thread_query_user_only_without_prior_question():
    msgs = [
        HumanMessage(content="hi"),
        AIMessage(content="Shalom. Speak."),
        HumanMessage(content="tell me about memory systems"),
    ]
    q = retrieval_thread_query(msgs)
    assert q == "tell me about memory systems"


def test_last_agent_question_skips_tool_only_turn():
    msgs = [
        HumanMessage(content="q"),
        AIMessage(
            content="",
            tool_calls=[{"name": "search_memory", "id": "c1", "args": {}}],
        ),
        HumanMessage(content="follow up?"),
    ]
    assert last_agent_question_before_last_human(msgs) is None
