"""Prune idle per-key locks so long-running bots do not grow maps forever."""

from __future__ import annotations

from typing import TypeVar

K = TypeVar("K")
L = TypeVar("L")


def _lock_busy(lock: L) -> bool:
    locked = getattr(lock, "locked", None)
    if callable(locked):
        return bool(locked())
    return False


def prune_unlocked_locks(
    locks: dict[K, L],
    *,
    max_size: int = 512,
) -> int:
    """Drop unlocked entries when *locks* exceeds *max_size*. Returns keys removed."""
    if len(locks) <= max_size:
        return 0
    stale = [key for key, lock in list(locks.items()) if not _lock_busy(lock)]
    removed = 0
    for key in stale:
        if len(locks) <= max_size:
            break
        if key in locks and not _lock_busy(locks[key]):
            locks.pop(key, None)
            removed += 1
    return removed
