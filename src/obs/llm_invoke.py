"""Wrapped LLM invoke/ainvoke with OTel LLM spans."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from src.connect_errors import is_transient_connection_error
from src.obs.llm_span import (
    annotate_llm_response,
    llm_call_span,
    metadata_from_response,
)
from src.obs.logs import log_event
from src.obs.metrics import record_llm_tokens
from src.obs.tracing import is_telemetry_enabled
from src.utils import get_message_text

_LLM_CONNECT_RETRIES = 3
_LLM_CONNECT_BACKOFF_S = (0.5, 1.0, 2.0)


def _prompt_chars(messages: Any) -> int:
    if isinstance(messages, str):
        return len(messages)
    total = 0
    if not isinstance(messages, (list, tuple)):
        return 0
    for msg in messages:
        total += len(get_message_text(msg))
    return total


def _model_name(llm: Any) -> str:
    for attr in ("model", "model_name"):
        val = getattr(llm, attr, None)
        if val:
            return str(val)
    return "unknown"


def _response_text(response: Any) -> str:
    return get_message_text(response) if response is not None else ""


def _annotate_success(
    span: Any,
    *,
    role: str,
    llm: Any,
    response: Any,
) -> None:
    text = _response_text(response)
    ec, pec, fr = metadata_from_response(response)
    annotate_llm_response(
        span,
        response_text=text,
        eval_count=ec,
        prompt_eval_count=pec,
        finish_reason=fr,
    )
    record_llm_tokens(ec, pec)
    log_event(
        "llm.invoke",
        role=role,
        model=_model_name(llm),
        input_tokens=int(pec or 0),
        output_tokens=int(ec or 0),
        finish_reason=fr or "",
    )


def _note_length_cut(role: str, response: Any) -> None:
    """Always log length cuts — OTEL is off by default."""
    ec, _pec, fr = metadata_from_response(response)
    if not fr or str(fr).lower() != "length":
        return
    log_event(
        "llm.length_cut",
        role=role,
        output_tokens=int(ec or 0),
    )
    from src.obs.metrics import record_llm_length_cut

    record_llm_length_cut()


def _background_cancellable() -> bool:
    """Traffic light, yellow: background LLM calls stream so a live turn can
    abort them mid-generation (closing the stream stops Ollama in seconds)."""
    try:
        from src.reverie.background import is_reverie_background_pass

        return is_reverie_background_pass()
    except Exception:
        return False


def _stream_with_cancel(llm: Any, messages: Any, *, config: dict[str, Any] | None):
    from src.reverie.cancel import ReverieCancelledError, is_reverie_cancel_requested

    if is_reverie_cancel_requested():
        raise ReverieCancelledError("live turn preempted background LLM call")
    stream = (
        llm.stream(messages, config=config)
        if config is not None
        else llm.stream(messages)
    )
    agg = None
    try:
        for chunk in stream:
            agg = chunk if agg is None else agg + chunk
            if is_reverie_cancel_requested():
                raise ReverieCancelledError("live turn preempted background LLM call")
    finally:
        try:
            close = getattr(stream, "close", None)
            if callable(close):
                close()
        except Exception:
            pass  # abort path — never mask the cancel
    if agg is None:
        raise RuntimeError("empty stream from llm")
    from langchain_core.messages import BaseMessageChunk, message_chunk_to_message

    # Finalize chunk → message: checkpoint serde and reducers expect AIMessage.
    if isinstance(agg, BaseMessageChunk):
        return message_chunk_to_message(agg)
    return agg


def _call_llm(llm: Any, messages: Any, *, config: dict[str, Any] | None) -> Any:
    if _background_cancellable() and hasattr(llm, "stream"):
        return _stream_with_cancel(llm, messages, config=config)
    if config is not None:
        return llm.invoke(messages, config=config)
    return llm.invoke(messages)


def _invoke_once(
    llm: Any,
    messages: Any,
    *,
    role: str,
    config: dict[str, Any] | None,
) -> Any:
    if not is_telemetry_enabled():
        response = _call_llm(llm, messages, config=config)
        _note_length_cut(role, response)
        return response

    with llm_call_span(
        model=_model_name(llm),
        role=role,
        prompt_chars=_prompt_chars(messages),
    ) as span:
        response = _call_llm(llm, messages, config=config)
        _annotate_success(span, role=role, llm=llm, response=response)
        _note_length_cut(role, response)
        return response


async def _ainvoke_once(
    llm: Any,
    messages: Any,
    *,
    role: str,
    config: dict[str, Any] | None,
) -> Any:
    if not is_telemetry_enabled():
        if config is not None:
            response = await llm.ainvoke(messages, config=config)
        else:
            response = await llm.ainvoke(messages)
        _note_length_cut(role, response)
        return response

    with llm_call_span(
        model=_model_name(llm),
        role=role,
        prompt_chars=_prompt_chars(messages),
    ) as span:
        if config is not None:
            response = await llm.ainvoke(messages, config=config)
        else:
            response = await llm.ainvoke(messages)
        _annotate_success(span, role=role, llm=llm, response=response)
        _note_length_cut(role, response)
        return response


def invoke_llm(
    llm: Any,
    messages: Any,
    *,
    role: str,
    config: dict[str, Any] | None = None,
) -> Any:
    last_err: BaseException | None = None
    for attempt in range(_LLM_CONNECT_RETRIES):
        try:
            return _invoke_once(llm, messages, role=role, config=config)
        except BaseException as exc:
            last_err = exc
            if (
                not is_transient_connection_error(exc)
                or attempt + 1 >= _LLM_CONNECT_RETRIES
            ):
                raise
            wait = _LLM_CONNECT_BACKOFF_S[min(attempt, len(_LLM_CONNECT_BACKOFF_S) - 1)]
            log_event(
                "llm.connect_retry",
                role=role,
                attempt=attempt + 1,
                wait_s=wait,
                err=str(exc)[:160],
            )
            time.sleep(wait)
    if last_err is not None:
        raise last_err
    raise RuntimeError("llm invoke failed without exception")


async def ainvoke_llm(
    llm: Any,
    messages: Any,
    *,
    role: str,
    config: dict[str, Any] | None = None,
) -> Any:
    last_err: BaseException | None = None
    for attempt in range(_LLM_CONNECT_RETRIES):
        try:
            return await _ainvoke_once(llm, messages, role=role, config=config)
        except BaseException as exc:
            last_err = exc
            if (
                not is_transient_connection_error(exc)
                or attempt + 1 >= _LLM_CONNECT_RETRIES
            ):
                raise
            wait = _LLM_CONNECT_BACKOFF_S[min(attempt, len(_LLM_CONNECT_BACKOFF_S) - 1)]
            log_event(
                "llm.connect_retry",
                role=role,
                attempt=attempt + 1,
                wait_s=wait,
                err=str(exc)[:160],
            )
            await asyncio.sleep(wait)
    if last_err is not None:
        raise last_err
    raise RuntimeError("llm ainvoke failed without exception")


__all__ = ["ainvoke_llm", "invoke_llm"]
