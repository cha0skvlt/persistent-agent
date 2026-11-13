"""Two-lane retrieval: world facts + diary. No self-query lane."""

from __future__ import annotations

from unittest.mock import MagicMock

from langchain_core.messages import HumanMessage

from src.graph import retrieval_node as rn


def test_diary_lane_uses_default_top_k(monkeypatch):
    top_k_seen: list[int] = []

    class _FakeSR:
        def search_with_ids(self, query, top_k=3, agent=None, **kwargs):
            top_k_seen.append(top_k)
            return []

        def latest_with_ids(self, n=2, agent=None):
            raise AssertionError("latest_with_ids must not run: no self-query lane")

    fake_mem = MagicMock()
    fake_mem.search.return_value = []
    monkeypatch.setattr(rn.settings, "self_reflection_retrieve_top_k", 4, raising=False)
    monkeypatch.setattr(rn, "_get_mem", lambda: fake_mem, raising=True)
    monkeypatch.setattr(
        rn, "get_default_self_reflection", lambda: _FakeSR(), raising=True
    )
    rn.retrieve_memory(  # type: ignore[arg-type]
        {"messages": [HumanMessage(content="What is the weather in Lisbon?")]}
    )

    assert top_k_seen == [4]
