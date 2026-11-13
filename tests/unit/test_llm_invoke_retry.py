"""LLM invoke retries on transient Ollama connect failures."""

from __future__ import annotations

import httpx
from langchain_core.messages import HumanMessage


def test_invoke_llm_retries_connect_error(monkeypatch):
    from src.obs import llm_invoke as li

    calls = {"n": 0}

    class _LLM:
        model = "test"

        def invoke(self, messages, config=None):
            calls["n"] += 1
            if calls["n"] < 2:
                raise httpx.ConnectError("All connection attempts failed")
            return HumanMessage(content="ok")

    monkeypatch.setattr(li, "is_telemetry_enabled", lambda: False)
    monkeypatch.setattr(li, "time", __import__("time"))
    slept: list[float] = []
    monkeypatch.setattr(li.time, "sleep", lambda s: slept.append(s))

    out = li.invoke_llm(_LLM(), [HumanMessage(content="hi")], role="agent")
    assert getattr(out, "content", "") == "ok"
    assert calls["n"] == 2
    assert slept


def test_invoke_llm_logs_length_cut(monkeypatch):
    from langchain_core.messages import AIMessage

    from src.obs import llm_invoke as li

    events: list[str] = []

    class _LLM:
        model = "test"

        def invoke(self, messages, config=None):
            return AIMessage(
                content="cut",
                response_metadata={"done_reason": "length", "eval_count": 2048},
            )

    monkeypatch.setattr(li, "is_telemetry_enabled", lambda: False)
    monkeypatch.setattr(
        li, "log_event", lambda ev, **_k: events.append(ev), raising=True
    )
    li.invoke_llm(_LLM(), [HumanMessage(content="hi")], role="agent")
    assert "llm.length_cut" in events
