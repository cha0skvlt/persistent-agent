"""Integration: Mind Hub injected via retrieve_memory."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from langchain_core.messages import HumanMessage

from src.config import settings
from src.graph.retrieval_node import retrieve_memory
from src.state.agent_state import AgentState


@pytest.mark.integration
def test_retrieve_memory_includes_mind_timeline(monkeypatch):
    monkeypatch.setattr(settings, "mind_hub_enabled", True, raising=False)
    timeline = ["[2026-07-04 12:00] OUTREACH SENT (telegram): hello USER"]
    state: AgentState = {
        "messages": [HumanMessage(content="what did you tell me?")],
        "retrieved_memories": [],
        "current_time": "",
        "session_summary": "",
        "session_message_offset": 0,
        "turn_count": 1,
    }
    with patch("src.graph.retrieval_node.fetch_mind_timeline", return_value=timeline):
        with patch("src.graph.retrieval_node._get_mem") as get_mem:
            mem = get_mem.return_value
            mem.search.return_value = []
            mem.fetch_memory_embeddings_by_ids.return_value = {}
            with patch("src.graph.retrieval_node.get_default_self_reflection") as sr:
                sr.return_value.search_with_ids.return_value = []
                out = retrieve_memory(state)
    assert out.get("retrieved_mind_timeline") == timeline
