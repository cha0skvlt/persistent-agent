"""World-fact lane labels (USER vs AGENT) — deixis guard for ``memories``."""

from __future__ import annotations

import re

LABELED_FACT_PREFIX_RE = re.compile(r"^(USER|AGENT):\s", re.IGNORECASE)

# First-person in world lane without USER:/AGENT: — likely diary leak or deixis confusion.
_FIRST_PERSON_WORLD_FACT_RE = re.compile(
    r"(?ix)"
    r"(?:^|\s|\[)"
    r"(?:"
    r"i(?:'m|'ve|'ll|\s+am|\s+was|\s+have|\s+think|\s+want|\s+need|\s+feel)?(?:\s|$|\b)|"
    r"my(?:\s|$|\b)|"
    r"me(?:\s|$|\b)"
    r")"
)


def has_world_fact_label(content: str) -> bool:
    return bool(LABELED_FACT_PREFIX_RE.match((content or "").strip()))


def world_fact_label_error(content: str) -> str | None:
    """Return error token if ``content`` must not go to world ``memories`` as-is."""
    t = (content or "").strip()
    if not t:
        return "empty"
    if has_world_fact_label(t):
        return None
    if _FIRST_PERSON_WORLD_FACT_RE.search(t):
        return "prefix_required"
    return None


__all__ = [
    "LABELED_FACT_PREFIX_RE",
    "has_world_fact_label",
    "world_fact_label_error",
]
