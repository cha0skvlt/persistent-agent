"""Patch agent LLM factory so tests never hit Ollama unless @pytest.mark.live."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest


def patch_agent_llm(monkeypatch: pytest.MonkeyPatch, invoker: Any) -> None:
    """Route workflow agent_node LLM calls to a test double."""
    import src.llm_factory as lf

    lf.clear_llm_cache()
    monkeypatch.setattr(lf, "get_llm", lambda: invoker, raising=True)
    monkeypatch.setattr(lf, "get_agent_llm", lambda: invoker, raising=True)
    monkeypatch.setattr(lf, "get_llm_with_tools", lambda _tools: invoker, raising=True)
    monkeypatch.setattr(
        lf, "get_agent_llm_with_tools", lambda _tools: invoker, raising=True
    )


def patch_agent_llm_split(
    monkeypatch: pytest.MonkeyPatch, plain: Any, with_tools: Any
) -> None:
    """Plain LLM for no-tools path; separate invoker when tools are bound."""
    import src.llm_factory as lf

    lf.clear_llm_cache()
    monkeypatch.setattr(lf, "get_llm", lambda: plain, raising=True)
    monkeypatch.setattr(lf, "get_agent_llm", lambda: plain, raising=True)
    monkeypatch.setattr(
        lf, "get_llm_with_tools", lambda _tools: with_tools, raising=True
    )
    monkeypatch.setattr(
        lf, "get_agent_llm_with_tools", lambda _tools: with_tools, raising=True
    )


def patch_agent_llm_tools_callback(
    monkeypatch: pytest.MonkeyPatch,
    invoker: Any,
    on_bind_tools: Callable[[list[Any]], Any],
) -> None:
    """Invoker for plain path; callback receives tool list when tools are bound."""
    import src.llm_factory as lf

    def _with_tools(tools: list[Any]) -> Any:
        on_bind_tools(tools)
        return invoker

    lf.clear_llm_cache()
    monkeypatch.setattr(lf, "get_llm", lambda: invoker, raising=True)
    monkeypatch.setattr(lf, "get_agent_llm", lambda: invoker, raising=True)
    monkeypatch.setattr(lf, "get_llm_with_tools", _with_tools, raising=True)
    monkeypatch.setattr(lf, "get_agent_llm_with_tools", _with_tools, raising=True)
