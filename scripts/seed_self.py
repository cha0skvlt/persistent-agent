#!/usr/bin/env python3
"""Seed the canonical first-person self-reflection."""

from __future__ import annotations

from src.config import settings
from src.kg.entities import resolve_entity
from src.memory import AgentMemory, SelfReflection
from src.speaker import speaker_label

SEED_CONTENT = "I am."
SEED_IMPORTANCE = 1.0


def ensure_agent_entity() -> None:
    """Ensure the agent entity exists in ``entities``."""
    anchor = speaker_label()
    mem = AgentMemory()
    eid = resolve_entity(
        anchor,
        mem=mem,
        agent=settings.default_agent_id,
        canonical_only=True,
    )
    print(f"  ENTITY: {anchor!r} id={eid}")


def seed_self() -> None:
    """Insert ``I am.`` when absent; no-op when row already exists."""
    agent = settings.default_agent_id
    print(f"Seeding self_reflection for agent_id={agent!r}...")
    ensure_agent_entity()
    sr = SelfReflection()

    if sr.has_exact(SEED_CONTENT, agent=agent):
        print(f"  SKIP: {SEED_CONTENT!r} already present")
        return

    if sr.save(SEED_CONTENT, importance=SEED_IMPORTANCE, agent=agent):
        print(f"  SAVED: {SEED_CONTENT!r} importance={SEED_IMPORTANCE}")
    else:
        print("  SKIP: save returned false (dedup or injection guard)")


if __name__ == "__main__":
    seed_self()
