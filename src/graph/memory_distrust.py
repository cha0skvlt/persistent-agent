"""Retrieve-side memory distrust — score multipliers, not truth scrubbing.

``[MEM]`` candidates are evidence, not oracle facts. Downrank unverified,
aged, and (via importance demotion on write) contradicted rows before top-k.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def _as_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def age_decay_factor(
    created_at: datetime | None,
    *,
    half_life_days: float,
    now: datetime | None = None,
) -> float:
    """Exponential decay by age. ``half_life_days <= 0`` disables (returns 1.0).

    Missing ``created_at`` is treated as one half-life old (factor 0.5) — fail closed
    vs trusting undated rows as fresh.
    """
    hl = float(half_life_days)
    if hl <= 0.0:
        return 1.0
    created = _as_utc(created_at)
    if created is None:
        return 0.5
    ref = _as_utc(now) or datetime.now(timezone.utc)
    age_days = max(0.0, (ref - created).total_seconds() / 86400.0)
    # score *= 0.5 ** (age / half_life)
    return float(0.5 ** (age_days / hl))


def distrust_multiplier(
    memory_type: str | None,
    created_at: datetime | None,
    *,
    unverified_factor: float,
    half_life_days: float,
    now: datetime | None = None,
) -> float:
    """Combined retrieve distrust factor in ``[0, 1]`` (identity skips age decay).

    Empty / missing type is treated as unverified.
    """
    mtype = (memory_type or "").strip().lower() or "unverified"
    factor = 1.0
    if mtype == "unverified":
        uf = float(unverified_factor)
        if 0.0 < uf < 1.0:
            factor *= uf
        elif uf <= 0.0:
            factor *= 0.01
    if mtype != "identity":
        factor *= age_decay_factor(created_at, half_life_days=half_life_days, now=now)
    return max(0.0, min(1.0, factor))


# Material score hit for obs (ignore trivial age dust under ~half-life*0.07).
_MATERIAL_DISTRUST_MULT = 0.95


def apply_score_distrust(
    ranked: list[tuple[str, str, float, float]],
    meta: dict[str, tuple[str, Any]],
    *,
    unverified_factor: float,
    half_life_days: float,
    now: datetime | None = None,
) -> tuple[list[tuple[str, str, float, float]], list[str]]:
    """Multiply final scores by distrust; re-sort. Returns (ranked, demoted_ids)."""
    if not ranked:
        return ranked, []
    demoted: list[str] = []
    adjusted: list[tuple[str, str, float, float]] = []
    for mid, content, score, qsim in ranked:
        mid_s = str(mid)
        if mid_s not in meta:
            mtype, created = "unverified", None
        else:
            mtype, created = meta[mid_s]
            if not (mtype or "").strip():
                mtype = "unverified"
        mult = distrust_multiplier(
            mtype,
            created if isinstance(created, datetime) else None,
            unverified_factor=unverified_factor,
            half_life_days=half_life_days,
            now=now,
        )
        adj = float(score) * mult
        if mult < _MATERIAL_DISTRUST_MULT:
            demoted.append(str(mid))
        adjusted.append((mid, content, adj, qsim))
    adjusted.sort(key=lambda row: -row[2])
    return adjusted, demoted


def parse_retrieve_memory_types(raw: str | None) -> list[str]:
    """Comma-separated ``MEMORY_RETRIEVE_TYPES`` → non-empty type list."""
    text = (raw or "").strip()
    if not text:
        return ["semantic"]
    out: list[str] = []
    seen: set[str] = set()
    for part in text.split(","):
        t = part.strip().lower()
        if not t or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out or ["semantic"]


__all__ = [
    "age_decay_factor",
    "apply_score_distrust",
    "distrust_multiplier",
    "parse_retrieve_memory_types",
]
