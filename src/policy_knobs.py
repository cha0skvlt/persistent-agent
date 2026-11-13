"""Tiny durable policy store — decision knobs that change runtime behavior."""

from __future__ import annotations

import logging
import math
import threading
from uuid import UUID

from src.config import settings
from src.db.pool import get_pool
from src.obs.logs import log_event

_log = logging.getLogger(__name__)
_default: PolicyKnobs | None = None
_lock = threading.Lock()

KNOWN_KNOBS = frozenset(
    {
        "unverified_rerank_factor",
        "age_half_life_days",
        "outreach_initiative",
        "tool_curiosity",
        "hygiene_debt",
        "attention_reserve",
    }
)

_DEFAULTS: dict[str, float] = {
    "unverified_rerank_factor": 0.35,
    "age_half_life_days": 180.0,
    "outreach_initiative": 0.5,
    "tool_curiosity": 0.5,
    "hygiene_debt": 0.0,
    # 0.3 → defer solitary ~3 min after last live USER turn.
    "attention_reserve": 0.3,
}

# Clamp poisoned / non-finite DB values so retrieve distrust cannot die.
_KNOB_BOUNDS: dict[str, tuple[float, float]] = {
    "unverified_rerank_factor": (0.01, 1.0),
    "age_half_life_days": (1.0, 730.0),
    # Lower bound is the outreach gate floor, not 0.0: below it `message_user`
    # refuses, and refusing blocks the only event that raises the knob again.
    # A value under the floor is therefore an absorbing state — permanent
    # muteness — so the knob is not allowed to reach it.
    "outreach_initiative": (0.2, 1.0),
    "tool_curiosity": (0.0, 1.0),
    "hygiene_debt": (0.0, 5.0),
    "attention_reserve": (0.0, 1.0),
}


def sanitize_knob_value(key: str, value: object) -> float:
    """Finite clamp into known bounds; unknown/poison → default."""
    default = float(_DEFAULTS.get(key, 0.0))
    try:
        v = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    if not math.isfinite(v):
        return default
    lo, hi = _KNOB_BOUNDS.get(key, (default, default))
    return max(lo, min(hi, v))


# Below this → message_user refused (low initiative).
_OUTREACH_FLOOR = 0.2
# At/above → +1 tool round on background passes.
_CURIOSITY_EXTRA_ROUND = 0.65
# At/above → +2 essay tool rounds on solitary (distill path).
_ESSAY_CURIOSITY_FLOOR = 0.5
# attention_reserve * this = minutes of live-busy solitary defer.
# Keep scale modest — high scale + outcome nudges starved the dugout.
_ATTENTION_DEFER_SCALE_MIN = 5.0
_ATTENTION_DEFER_FLOOR_MIN = 0.5
_ATTENTION_DEFER_CEIL_MIN = 3.0


def get_default_policy_knobs() -> PolicyKnobs:
    global _default
    if _default is None:
        with _lock:
            if _default is None:
                _default = PolicyKnobs()
    return _default


def reset_policy_knobs_for_tests() -> None:
    global _default
    with _lock:
        _default = None


class PolicyKnobs:
    def __init__(self, pool=None):
        self.pool = pool or get_pool()

    def get_all(self, *, agent: str | None = None) -> dict[str, float]:
        aid = settings.default_agent_id if agent is None else agent
        out = dict(_DEFAULTS)
        out["unverified_rerank_factor"] = float(
            settings.memory_unverified_rerank_factor
        )
        out["age_half_life_days"] = float(settings.memory_age_half_life_days)
        try:
            with self.pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT key, value
                        FROM policy_knobs
                        WHERE agent_id = %s
                        """,
                        (aid,),
                    )
                    for key, value in cur.fetchall() or []:
                        k = str(key or "").strip()
                        if k in KNOWN_KNOBS:
                            out[k] = sanitize_knob_value(k, value)
        except Exception as exc:
            _log.warning("policy_knobs get_all failed: %s", exc)
        for k in KNOWN_KNOBS:
            out[k] = sanitize_knob_value(k, out.get(k, _DEFAULTS[k]))
        return out

    def set(
        self,
        key: str,
        value: float,
        *,
        agent: str | None = None,
        lesson_id: str | None = None,
    ) -> bool:
        k = (key or "").strip()
        if k not in KNOWN_KNOBS:
            return False
        aid = settings.default_agent_id if agent is None else agent
        val = sanitize_knob_value(k, value)
        lid: str | None = None
        if lesson_id:
            try:
                lid = str(UUID(str(lesson_id).strip()))
            except ValueError:
                lid = None
        try:
            with self.pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO policy_knobs (agent_id, key, value, lesson_id)
                        VALUES (%s, %s, %s, %s::uuid)
                        ON CONFLICT (agent_id, key) DO UPDATE
                        SET value = EXCLUDED.value,
                            lesson_id = COALESCE(EXCLUDED.lesson_id, policy_knobs.lesson_id),
                            updated_at = now()
                        """,
                        (aid, k, val, lid),
                    )
                conn.commit()
            log_event("policy.knob_set", key=k, value=val, lesson_id=lid)
            return True
        except Exception as exc:
            _log.warning("policy_knobs set failed: %s", exc)
            return False

    def apply_outcome(
        self,
        signal: str,
        *,
        agent: str | None = None,
        lesson_id: str | None = None,
    ) -> dict[str, float]:
        """Nudge knobs from a cheap outcome signal. Returns updated snapshot."""
        knobs = self.get_all(agent=agent)
        sig = (signal or "").strip().lower()
        changed: dict[str, float] = {}

        def _clamp(v: float, lo: float, hi: float) -> float:
            return max(lo, min(hi, v))

        if sig == "log_pile":
            changed["unverified_rerank_factor"] = _clamp(
                knobs["unverified_rerank_factor"] * 0.9, 0.1, 1.0
            )
            changed["age_half_life_days"] = _clamp(
                knobs["age_half_life_days"] * 0.95, 30.0, 730.0
            )
            changed["hygiene_debt"] = _clamp(
                float(knobs.get("hygiene_debt", 0.0)) + 1.0, 0.0, 5.0
            )
        elif sig == "hygiene_done":
            # Close the P2 loop: burn one debt unit and drift trust toward baselines.
            baseline_u = float(settings.memory_unverified_rerank_factor)
            baseline_a = float(settings.memory_age_half_life_days)
            changed["hygiene_debt"] = _clamp(
                float(knobs.get("hygiene_debt", 0.0)) - 1.0, 0.0, 5.0
            )
            cur_u = float(knobs["unverified_rerank_factor"])
            if cur_u < baseline_u:
                changed["unverified_rerank_factor"] = _clamp(
                    cur_u + 0.03, 0.1, baseline_u
                )
            cur_a = float(knobs["age_half_life_days"])
            if cur_a < baseline_a:
                changed["age_half_life_days"] = _clamp(cur_a * 1.05, 30.0, baseline_a)
        elif sig == "lesson_saved":
            changed["tool_curiosity"] = _clamp(knobs["tool_curiosity"] + 0.02, 0.0, 1.0)
        elif sig == "outreach_fulfilled":
            changed["outreach_initiative"] = _clamp(
                knobs["outreach_initiative"] + 0.05, 0.0, 1.0
            )
        elif sig == "attention_deferred":
            # Yielding to live USER is correct — do NOT widen the defer window
            # (raising reserve caused a skip_live_busy death spiral).
            changed["attention_reserve"] = _clamp(
                float(knobs.get("attention_reserve", 0.3)) - 0.01, 0.1, 1.0
            )

        applied: list[str] = []
        for key, value in changed.items():
            if self.set(key, value, agent=agent, lesson_id=lesson_id):
                applied.append(key)
        if applied:
            log_event("policy.outcome", signal=sig, keys=sorted(applied))
        return self.get_all(agent=agent)


def effective_knobs(*, agent: str | None = None) -> dict[str, float]:
    """One DB read for all knobs (prefer this over repeated effective_*)."""
    return get_default_policy_knobs().get_all(agent=agent)


def effective_unverified_factor(knobs: dict[str, float] | None = None) -> float:
    data = knobs if knobs is not None else effective_knobs()
    return float(
        data.get(
            "unverified_rerank_factor",
            settings.memory_unverified_rerank_factor,
        )
    )


def effective_age_half_life_days(knobs: dict[str, float] | None = None) -> float:
    data = knobs if knobs is not None else effective_knobs()
    return float(data.get("age_half_life_days", settings.memory_age_half_life_days))


def outreach_initiative_allows(knobs: dict[str, float] | None = None) -> bool:
    data = knobs if knobs is not None else effective_knobs()
    return float(data.get("outreach_initiative", 0.5)) >= _OUTREACH_FLOOR


def curiosity_extra_tool_rounds(knobs: dict[str, float] | None = None) -> int:
    data = knobs if knobs is not None else effective_knobs()
    if float(data.get("tool_curiosity", 0.5)) >= _CURIOSITY_EXTRA_ROUND:
        return 1
    return 0


def hygiene_debt(knobs: dict[str, float] | None = None) -> float:
    data = knobs if knobs is not None else effective_knobs()
    return float(data.get("hygiene_debt", 0.0))


def live_busy_defer_minutes(knobs: dict[str, float] | None = None) -> float:
    """Minutes after a live USER turn during which solitary must defer."""
    data = knobs if knobs is not None else effective_knobs()
    reserve = float(data.get("attention_reserve", 0.3))
    return max(
        _ATTENTION_DEFER_FLOOR_MIN,
        min(_ATTENTION_DEFER_CEIL_MIN, reserve * _ATTENTION_DEFER_SCALE_MIN),
    )


def solitary_deferred_for_live_busy(
    knobs: dict[str, float] | None = None,
) -> bool:
    """True when attention_reserve says live USER still owns the channel."""
    try:
        from src.reverie.idle import minutes_since_last_user_turn

        elapsed = minutes_since_last_user_turn()
    except Exception as exc:
        _log.warning("solitary live-busy clock failed: %s", exc)
        # Fail closed — defer solitary when idle clock is unreadable.
        return True
    if elapsed is None:
        return False
    return float(elapsed) < live_busy_defer_minutes(knobs)


def essay_extra_tool_rounds(knobs: dict[str, float] | None = None) -> int:
    """Solitary essay slack: +2 rounds when curiosity is warm."""
    try:
        from src.reverie.turn_kind import is_solitary_turn

        if not is_solitary_turn():
            return 0
    except Exception as exc:
        _log.warning("essay curiosity check failed: %s", exc)
        return 0
    data = knobs if knobs is not None else effective_knobs()
    if float(data.get("tool_curiosity", 0.5)) >= _ESSAY_CURIOSITY_FLOOR:
        return 2
    return 0


def apply_outcome_safe(signal: str, *, lesson_id: str | None = None) -> None:
    """Best-effort knob nudge; log failures, never raise into the hot path."""
    try:
        get_default_policy_knobs().apply_outcome(signal, lesson_id=lesson_id)
    except Exception as exc:
        _log.warning("policy apply_outcome(%s) failed: %s", signal, exc)


__all__ = [
    "KNOWN_KNOBS",
    "PolicyKnobs",
    "apply_outcome_safe",
    "curiosity_extra_tool_rounds",
    "effective_age_half_life_days",
    "effective_knobs",
    "effective_unverified_factor",
    "essay_extra_tool_rounds",
    "get_default_policy_knobs",
    "hygiene_debt",
    "live_busy_defer_minutes",
    "outreach_initiative_allows",
    "reset_policy_knobs_for_tests",
    "sanitize_knob_value",
    "solitary_deferred_for_live_busy",
]
