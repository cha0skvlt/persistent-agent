"""LangGraph node span helpers."""

from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Iterator, TypeVar

from src.obs.tracing import get_tracer, is_telemetry_enabled

tracer = get_tracer("agent.graph")

F = TypeVar("F", bound=Callable[..., Any])

_STATE_KEYS = ("needs_reflect", "session_summary_due")


@contextmanager
def traced_agent_turn(
    *,
    thread_id: str,
    channel: str,
    concurrent_deferred_job: bool,
) -> Iterator[None]:
    """Root span for one live user turn (overlap flag for deferred-job correlation)."""
    if not is_telemetry_enabled():
        yield
        return
    with tracer.start_as_current_span("agent.turn") as span:
        span.set_attribute("agent.thread_id", thread_id or "")
        span.set_attribute("agent.channel", channel or "unknown")
        span.set_attribute(
            "agent.concurrent_deferred_job", bool(concurrent_deferred_job)
        )
        try:
            yield
        except Exception as exc:
            span.record_exception(exc)
            raise


def traced_node(node_name: str) -> Callable[[F], F]:
    """Wrap a graph node so each invocation emits ``node.{name}`` span."""

    def deco(fn: F) -> F:
        @wraps(fn)
        def wrapper(state: Any, *args: Any, **kwargs: Any) -> Any:
            if not is_telemetry_enabled():
                return fn(state, *args, **kwargs)
            with tracer.start_as_current_span(f"node.{node_name}") as span:
                if isinstance(state, dict):
                    span.set_attribute(
                        "agent.turn_count", int(state.get("turn_count", 0) or 0)
                    )
                span.set_attribute("agent.thread_id", str(state.get("thread_id") or ""))
                try:
                    result = fn(state, *args, **kwargs)
                except Exception as exc:
                    span.record_exception(exc)
                    raise
                if isinstance(result, dict):
                    for key in _STATE_KEYS:
                        if key in result:
                            span.set_attribute(f"agent.{key}", bool(result[key]))
                return result

        return wrapper  # type: ignore[return-value]

    return deco


__all__ = ["traced_agent_turn", "traced_node"]
