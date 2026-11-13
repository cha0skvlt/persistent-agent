"""LLM call spans with GenAI semantic conventions."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from opentelemetry.trace import Span

from src.obs.tracing import get_tracer

tracer = get_tracer("agent.llm")


@contextmanager
def llm_call_span(*, model: str, role: str, prompt_chars: int) -> Iterator[Span]:
    """``role``: agent | reflect | summarize | solitary | …"""
    with tracer.start_as_current_span(f"llm.{role}") as span:
        span.set_attribute("gen_ai.system", "ollama")
        span.set_attribute("gen_ai.request.model", model)
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("agent.llm.role", role)
        span.set_attribute("agent.llm.prompt_chars", int(prompt_chars))
        yield span


def annotate_llm_response(
    span: Span,
    *,
    response_text: str,
    eval_count: int | None,
    prompt_eval_count: int | None,
    finish_reason: str | None,
) -> None:
    if eval_count is not None:
        span.set_attribute("gen_ai.usage.output_tokens", int(eval_count))
    if prompt_eval_count is not None:
        span.set_attribute("gen_ai.usage.input_tokens", int(prompt_eval_count))
    if finish_reason:
        span.set_attribute("gen_ai.response.finish_reasons", [finish_reason])
    span.set_attribute("agent.llm.response_chars", len(response_text or ""))


def metadata_from_response(response: Any) -> tuple[int | None, int | None, str | None]:
    """Extract Ollama token stats from LangChain ``AIMessage.response_metadata``."""
    meta = getattr(response, "response_metadata", None) or {}
    if not isinstance(meta, dict):
        return None, None, None
    eval_count = meta.get("eval_count")
    prompt_eval_count = meta.get("prompt_eval_count")
    finish = meta.get("done_reason") or meta.get("finish_reason")
    try:
        ec = int(eval_count) if eval_count is not None else None
    except (TypeError, ValueError):
        ec = None
    try:
        pec = int(prompt_eval_count) if prompt_eval_count is not None else None
    except (TypeError, ValueError):
        pec = None
    fr = str(finish) if finish is not None else None
    return ec, pec, fr


__all__ = [
    "annotate_llm_response",
    "llm_call_span",
    "metadata_from_response",
]
