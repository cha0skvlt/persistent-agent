"""Canonical KG predicate whitelist (curation-only graph)."""

from __future__ import annotations

ALLOWED_PREDICATES = {
    # identity / being
    "is",
    "was",
    "alias_of",
    "embodies",
    "manifests",
    # hierarchical / structural
    "contains",
    "contained_in",
    "has",
    "part_of",
    # opposition / relation
    "opposed_to",
    "requires",
    "themes",
    # personal relations
    "loves",
    "hates",
    "knows",
    "friend_of",
    "enemy_of",
    # kinship
    "son_of",
    "daughter_of",
    "father_of",
    "mother_of",
    "married_to",
    # geography / work
    "born_in",
    "lives_in",
    "works_as",
    # practice / study
    "read",
    "practices",
    "studied",
    # infrastructure
    "runs_on",
    "hosts",
}


def is_allowed(predicate: str) -> bool:
    return (predicate or "").strip().lower() in ALLOWED_PREDICATES


__all__ = ["ALLOWED_PREDICATES", "is_allowed"]
