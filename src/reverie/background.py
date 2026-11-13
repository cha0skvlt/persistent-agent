"""Mark reverie graph passes so live-only tools stay off the background path.

Implementation lives in ``turn_kind`` — this module re-exports the compat API.
"""

from __future__ import annotations

from src.reverie.turn_kind import (
    intent_background_pass,
    is_intent_background_pass,
    is_reverie_background_pass,
    reverie_background_pass,
)

__all__ = [
    "is_reverie_background_pass",
    "reverie_background_pass",
    "is_intent_background_pass",
    "intent_background_pass",
]
