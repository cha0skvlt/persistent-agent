"""Shared test utilities (mock pools, graph state, tracking fakes)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from langchain_core.messages import HumanMessage


def mock_pool_with_cursor(cur: MagicMock) -> MagicMock:
    """Return a pool whose connection yields ``cur`` inside a transaction."""
    cur_ctx = MagicMock()
    cur_ctx.__enter__ = MagicMock(return_value=cur)
    cur_ctx.__exit__ = MagicMock(return_value=False)
    conn = MagicMock()
    conn.cursor.return_value = cur_ctx
    tx_ctx = MagicMock()
    tx_ctx.__enter__ = MagicMock(return_value=None)
    tx_ctx.__exit__ = MagicMock(return_value=False)
    conn.transaction.return_value = tx_ctx
    conn_ctx = MagicMock()
    conn_ctx.__enter__ = MagicMock(return_value=conn)
    conn_ctx.__exit__ = MagicMock(return_value=False)
    pool = MagicMock()
    pool.connection.return_value = conn_ctx
    return pool


def graph_state(human: str, /, **extra: Any) -> dict[str, Any]:
    """Minimal ``AgentState`` for route / retrieve unit tests."""
    base: dict[str, Any] = {
        "messages": [HumanMessage(content=human)],
        "retrieved_memories": [],
        "current_time": "",
        "session_summary": "",
        "session_message_offset": 0,
    }
    base.update(extra)
    return base


class TrackingSearchMem:
    """Fake ``AgentMemory`` that records ``search()`` kwargs (embed reuse tests)."""

    def __init__(self) -> None:
        self.search_calls: list[dict[str, Any]] = []

    def get_vector(self, text: str, **kw: Any) -> list[float]:
        return [0.1] * 8

    def search(self, query: str, **kw: Any) -> list:
        self.search_calls.append({"query": query, **kw})
        return []

    def fetch_memory_embeddings_by_ids(self, ids: list, **kw: Any) -> dict:
        return {}

    def fetch_memory_rerank_meta_by_ids(self, ids: list, **kw: Any) -> dict:
        return {str(i): ("semantic", None) for i in ids}

    def entities_matching_tokens(self, *a: Any, **kw: Any) -> list:
        return []

    def traverse_graph_batch(self, *a: Any, **kw: Any) -> dict:
        return {}
