"""LangGraph test helpers — fresh MemorySaver, thread config, seeded state."""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from src.agent_runner import graph_invoke_config, reset_graph_app
from src.config import settings
from src.graph import retrieval_node as rn
from src.graph import workflow as w
from src.graph.checkpoint_serde import get_checkpoint_serde
from src.graph.route_turn_node import route_turn_node
from tests.support.helpers import TrackingSearchMem, graph_state


def fresh_memory_graph():
    """Compile graph with an isolated in-memory checkpointer (per test)."""
    serde = get_checkpoint_serde()
    return w._wire_state_graph().compile(checkpointer=MemorySaver(serde=serde))


def fresh_thread_config(*, suffix: str = "") -> dict[str, Any]:
    """Unique ``thread_id`` + recursion limit — no checkpoint bleed between tests."""
    tid = suffix.strip() or f"test_{uuid.uuid4().hex}"
    return graph_invoke_config(thread_id=tid)


def initial_turn_input(
    human: str,
    *,
    session_uuid: str | None = None,
) -> dict[str, Any]:
    """Minimal invoke payload (matches ``agent_runner``)."""
    return {
        "messages": [HumanMessage(content=human)],
        "current_time": "",
        "session_uuid": session_uuid or uuid.uuid4().hex,
    }


def patch_retrieve_noop(monkeypatch: pytest.MonkeyPatch) -> TrackingSearchMem:
    """Stub ``retrieve_memory`` backends for fast graph tests."""
    mem = TrackingSearchMem()
    monkeypatch.setattr(rn, "_get_mem", lambda: mem, raising=True)
    monkeypatch.setattr(
        rn,
        "get_default_self_reflection",
        lambda: MagicMock(search=lambda *a, **k: []),
        raising=True,
    )
    return mem


def seed_post_retrieve_state(
    human: str,
    *,
    current_time: str = "2026-06-30T12:00:00",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """State after ``route_turn`` + ``retrieve_memory`` — resume at ``agent``."""
    state = graph_state(human, current_time=current_time)
    state.update(route_turn_node(state))
    state.update(rn.retrieve_memory(state))
    if extra:
        state.update(extra)
    return state


def graph_test_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Defaults for graph integration tests (memory checkpointer, no reflect)."""
    monkeypatch.setattr(settings, "checkpointer", "memory", raising=False)
    monkeypatch.setattr(settings, "reflection", False, raising=False)
    reset_graph_app()
