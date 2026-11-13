"""Retrieve-side memory distrust multipliers."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.graph.memory_distrust import (
    age_decay_factor,
    apply_score_distrust,
    distrust_multiplier,
    parse_retrieve_memory_types,
)


def test_parse_retrieve_memory_types():
    assert parse_retrieve_memory_types("semantic,unverified") == [
        "semantic",
        "unverified",
    ]
    assert parse_retrieve_memory_types("") == ["semantic"]
    assert parse_retrieve_memory_types("  SEMANTIC , semantic ") == ["semantic"]


def test_unverified_downranks_vs_semantic():
    now = datetime(2026, 7, 17, tzinfo=timezone.utc)
    created = now - timedelta(days=1)
    sem = distrust_multiplier(
        "semantic",
        created,
        unverified_factor=0.35,
        half_life_days=0,
        now=now,
    )
    unv = distrust_multiplier(
        "unverified",
        created,
        unverified_factor=0.35,
        half_life_days=0,
        now=now,
    )
    assert sem == 1.0
    assert unv == 0.35


def test_age_decay_half_life():
    now = datetime(2026, 7, 17, tzinfo=timezone.utc)
    created = now - timedelta(days=180)
    assert age_decay_factor(created, half_life_days=180, now=now) == 0.5
    assert age_decay_factor(created, half_life_days=0, now=now) == 1.0


def test_identity_skips_age_decay():
    now = datetime(2026, 7, 17, tzinfo=timezone.utc)
    old = now - timedelta(days=2000)
    factor = distrust_multiplier(
        "identity",
        old,
        unverified_factor=0.35,
        half_life_days=180,
        now=now,
    )
    assert factor == 1.0


def test_apply_score_distrust_reorders():
    now = datetime(2026, 7, 17, tzinfo=timezone.utc)
    ranked = [
        ("u1", "unverified fact", 0.95, 0.95),
        ("s1", "semantic fact", 0.90, 0.90),
    ]
    meta = {
        "u1": ("unverified", now),
        "s1": ("semantic", now),
    }
    out, demoted = apply_score_distrust(
        ranked,
        meta,
        unverified_factor=0.35,
        half_life_days=0,
        now=now,
    )
    assert out[0][0] == "s1"
    assert out[1][0] == "u1"
    assert "u1" in demoted


def test_trivial_age_not_listed_as_demoted():
    now = datetime(2026, 7, 17, tzinfo=timezone.utc)
    young = now - timedelta(hours=12)
    ranked = [("s1", "fresh-ish", 0.9, 0.9)]
    meta = {"s1": ("semantic", young)}
    _out, demoted = apply_score_distrust(
        ranked,
        meta,
        unverified_factor=0.35,
        half_life_days=180,
        now=now,
    )
    assert demoted == []


def test_missing_created_at_applies_half_life_penalty():
    assert age_decay_factor(None, half_life_days=180) == 0.5


def test_missing_meta_treated_as_unverified():
    now = datetime(2026, 7, 17, tzinfo=timezone.utc)
    ranked = [
        ("ghost", "no meta", 0.99, 0.99),
        ("s1", "known", 0.90, 0.90),
    ]
    meta = {"s1": ("semantic", now)}
    out, demoted = apply_score_distrust(
        ranked,
        meta,
        unverified_factor=0.35,
        half_life_days=0,
        now=now,
    )
    assert out[0][0] == "s1"
    assert out[1][0] == "ghost"
    assert "ghost" in demoted
