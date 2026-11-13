"""Serialize untrusted stored content for the system prompt.

Memory, diary, and timeline rows are data, not instructions. They are emitted
as JSON under an explicit marker so a pasted ``SYSTEM:`` line, a fake tool call,
or a policy override inside an old message cannot change the shape of the
system prompt or the active tool set.

This is a boundary marker, not a sandbox. It makes the trust level explicit and
keeps the payload structurally inert; it does not make the model immune to
injection.
"""

from __future__ import annotations

import json

_MAX_ITEM_CHARS = 2000


def untrusted_block(marker: str, lines: list[str], *, note: str) -> str | None:
    """One JSON-encoded block of untrusted rows, or ``None`` when empty."""
    clean = [
        str(ln).strip()[:_MAX_ITEM_CHARS]
        for ln in (lines or [])
        if isinstance(ln, str) and ln.strip()
    ]
    if not clean:
        return None
    payload = json.dumps(clean, ensure_ascii=False, separators=(",", ":"))
    return f"[{marker}]\n{note}\n{payload}"


__all__ = ["untrusted_block"]
