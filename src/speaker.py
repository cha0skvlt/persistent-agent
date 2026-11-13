"""Speaker labels and prefix patterns.

The agent has no configurable identity: one fixed label, one fixed subject key set.
"""

from __future__ import annotations

import re

SPEAKER = "Agent"

# Lowercased names that resolve to the local agent in KG / reflect guards.
_SUBJECT_KEYS = frozenset({"agent", "the agent", "assistant"})

# Display / vocative forms used in leaked speaker prefixes and rubric headers.
_SURFACE_FORMS = frozenset({"Agent"})


def speaker_label() -> str:
    """Transcript / reflect speaker label."""
    return SPEAKER


def agent_key() -> str:
    return SPEAKER.casefold()


def subject_keys() -> frozenset[str]:
    """Lowercased names treated as the local agent in reflect / KG guards."""
    return _SUBJECT_KEYS


def surface_forms() -> frozenset[str]:
    """Display forms used in leaked speaker prefixes and rubric headers."""
    return _SURFACE_FORMS


def is_generic_assistant_surface(name: str) -> bool:
    """True for generic assistant callsigns that resolve to the agent."""
    n = (name or "").strip().lower()
    return bool(n) and n in _SUBJECT_KEYS


def subject_is_agent(name: str) -> bool:
    """True when a KG triple subject names the agent."""
    n = (name or "").strip().lower()
    return bool(n) and n in _SUBJECT_KEYS


def address_prefix_pattern() -> re.Pattern[str]:
    """Strip leading agent/user vocatives from search queries."""
    parts = [re.escape(p) for p in ("Agent", "user", "agent", "hey", "listen")]
    inner = "|".join(dict.fromkeys(parts))
    return re.compile(
        rf"^(?:(?:{inner})[,\s:!—–-]+)+",
        re.IGNORECASE | re.UNICODE,
    )


def speaker_prefix_pattern() -> re.Pattern[str]:
    """Strip leaked ``Agent answers:`` wrappers."""
    return re.compile(
        r"^\s*(?:Agent\s+(?:answers|responds))\s*[:：]\s*[-—–]?\s*",
        re.IGNORECASE | re.UNICODE,
    )


def rubric_header_pattern() -> re.Pattern[str]:
    """Detect template rubric lines ``Agent — …`` at start of reply."""
    return re.compile(r"^Agent\s*[—–\-]", re.IGNORECASE | re.UNICODE)


__all__ = [
    "SPEAKER",
    "address_prefix_pattern",
    "agent_key",
    "is_generic_assistant_surface",
    "rubric_header_pattern",
    "speaker_label",
    "speaker_prefix_pattern",
    "subject_is_agent",
    "subject_keys",
    "surface_forms",
]
