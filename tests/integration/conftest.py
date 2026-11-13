"""Integration tier — graph pipelines and multi-module flows (mocked backends)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from src.config import settings
from src.graph import workflow as w
from tests.support.graph_fixtures import (
    fresh_memory_graph,
    fresh_thread_config,
    graph_test_settings,
    patch_retrieve_noop,
)


@pytest.fixture
def memory_graph_app(monkeypatch: pytest.MonkeyPatch):
    """Fresh ``MemorySaver`` graph — isolated checkpoint per test."""
    graph_test_settings(monkeypatch)
    return fresh_memory_graph()


@pytest.fixture
def graph_thread_config() -> dict[str, Any]:
    """Unique ``thread_id`` for checkpoint isolation."""
    return fresh_thread_config()


@pytest.fixture
def retrieve_mocks(monkeypatch: pytest.MonkeyPatch):
    """No-op memory retrieval for graph invoke tests."""
    return patch_retrieve_noop(monkeypatch)


@pytest.fixture
def mock_agent_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[AIMessage | str], None]:
    """Patch ``invoke_llm`` for ``agent_node`` / full-graph tests."""

    def _apply(reply: AIMessage | str, *, may_use_tools: bool = True) -> None:
        msg = reply if isinstance(reply, AIMessage) else AIMessage(content=reply)
        monkeypatch.setattr(w, "invoke_llm", lambda *a, **k: msg, raising=True)
        monkeypatch.setattr(
            w, "_agent_may_call_tools", lambda *a, **k: may_use_tools, raising=True
        )

    return _apply


@pytest.fixture
def solitary_test_settings(monkeypatch: pytest.MonkeyPatch, reset_solitary_state):
    """Solitary enabled with zero interval for integration tests."""
    from src.agent_runner import reset_llm_turn_semaphore

    monkeypatch.setattr(settings, "solitary_enabled", True, raising=False)
    monkeypatch.setattr(settings, "solitary_interval_min", 0, raising=False)
    # Live-busy defer reads the idle clock from DB — fail-closed would skip
    # every pass in tests; pretend the channel is free.
    monkeypatch.setattr(
        "src.policy_knobs.solitary_deferred_for_live_busy",
        lambda *a, **k: False,
    )
    reset_llm_turn_semaphore()
    graph_test_settings(monkeypatch)
    yield
    reset_llm_turn_semaphore()
