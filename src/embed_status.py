"""Per-turn embed availability (typed degradation — not silent empty search)."""

from __future__ import annotations

import contextvars

from src.obs.logs import log_event

_embed_unavailable: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "embed_unavailable", default=None
)


def reset_embed_status() -> None:
    """Clear degradation flag at the start of a retrieve turn."""
    _embed_unavailable.set(None)


def mark_embed_unavailable(detail: str) -> None:
    """Record that embeddings failed; search may return empty without inventing hits."""
    msg = (detail or "unavailable").strip()[:200] or "unavailable"
    _embed_unavailable.set(msg)
    log_event("embed.unavailable", detail=msg)


def embed_unavailable_detail() -> str | None:
    return _embed_unavailable.get()


def embed_unavailable_sys_line() -> str | None:
    detail = _embed_unavailable.get()
    if not detail:
        return None
    return (
        "[SYS] Embeddings unavailable this turn "
        f"({detail}). No memory/KG retrieve — do not invent past facts."
    )


__all__ = [
    "embed_unavailable_detail",
    "embed_unavailable_sys_line",
    "mark_embed_unavailable",
    "reset_embed_status",
]
