"""Per-turn node timings for Telegram footer (thread-local, one graph invoke)."""

from __future__ import annotations

import threading

from src.obs.logs import format_elapsed_ms

_tls = threading.local()


def reset_turn_timing() -> None:
    """Clear timings at the start of a user turn."""
    _tls.nodes = {}
    _tls.concurrent_deferred_job = False


def set_turn_concurrent_deferred(active: bool) -> None:
    """Record whether this live turn overlapped an in-flight deferred job."""
    _tls.concurrent_deferred_job = bool(active)


def turn_had_concurrent_deferred() -> bool:
    return bool(getattr(_tls, "concurrent_deferred_job", False))


def record_node_timing(node: str, elapsed_ms: float | int | None) -> None:
    if not node or elapsed_ms is None:
        return
    nodes: dict[str, float] = getattr(_tls, "nodes", None) or {}
    nodes[str(node)] = float(elapsed_ms)
    _tls.nodes = nodes


def get_turn_timings() -> dict[str, float]:
    """Snapshot node timings for the current turn (same thread as graph invoke)."""
    nodes: dict[str, float] = getattr(_tls, "nodes", None) or {}
    return dict(nodes)


def format_node_timings(nodes: dict[str, float] | None = None) -> str:
    """Compact ``agent=45s retrieve=0.5s`` for footer."""
    data: dict[str, float] = nodes if nodes is not None else get_turn_timings()
    if not data:
        return ""
    order = (
        "retrieve_memory",
        "summarize_session",
        "agent_node",
        "tools",
        "reflect_node",
        "update_time",
    )
    parts: list[str] = []
    seen: set[str] = set()
    for name in order:
        if name in data:
            parts.append(f"{name}={format_elapsed_ms(data[name])}")
            seen.add(name)
    for name in sorted(data):
        if name not in seen:
            parts.append(f"{name}={format_elapsed_ms(data[name])}")
    return "  ".join(parts)
