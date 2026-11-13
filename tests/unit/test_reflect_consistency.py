"""Deferred reflect consistency gate — contradiction detection and policies."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.graph import reflect_consistency as rc
from src.obs.logs import log_event, setup_agent_event_logger


@pytest.mark.unit
def test_high_similarity_threshold_optional_override(monkeypatch):
    monkeypatch.setattr(rc.settings, "reflect_consistency_gate_sim_threshold", 0.91)
    assert rc.high_similarity_threshold() == 0.91


@pytest.mark.unit
def test_high_similarity_threshold_derived_from_floor(monkeypatch):
    monkeypatch.setattr(rc.settings, "memory_similarity_floor", 0.3)
    assert rc.high_similarity_threshold() == pytest.approx(0.86)
    monkeypatch.setattr(rc.settings, "memory_similarity_floor", 0.5)
    assert rc.high_similarity_threshold() == pytest.approx(0.9)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("new_text", "existing_text", "expected"),
    [
        (
            "USER does not use Python for backend services",
            "USER uses Python for backend services",
            True,
        ),
        (
            "USER never deploys on Kubernetes clusters",
            "USER deploys on Kubernetes clusters regularly",
            True,
        ),
        (
            "USER prefers Rust for systems programming",
            "USER enjoys hiking on weekends",
            False,
        ),
        (
            "USER uses Python for backend services",
            "USER uses Python for backend services",
            False,
        ),
        (
            "USER does not use Python",
            "USER uses Python",
            True,
        ),
        (
            "USER does not use Python",
            "USER enjoys hiking",
            False,
        ),
    ],
)
def test_texts_semantically_contradict(new_text, existing_text, expected):
    assert rc.texts_semantically_contradict(new_text, existing_text) is expected


@pytest.mark.unit
def test_find_memory_contradiction_requires_high_sim_and_polarity(monkeypatch):
    monkeypatch.setattr(rc.settings, "memory_similarity_floor", 0.3)
    mem = MagicMock()
    new = "USER does not use Python for backend services"
    existing = "USER uses Python for backend services"

    mem.search.return_value = [("id-1", existing, 0.95)]
    hit = rc.find_memory_contradiction(mem, new)
    assert hit == ("id-1", existing, 0.95)
    mem.search.assert_called_once()
    assert mem.search.call_args.kwargs.get("memory_types") == [
        "semantic",
        "unverified",
    ]

    mem.search.return_value = [("id-1", existing, 0.5)]
    assert rc.find_memory_contradiction(mem, new) is None

    mem.search.return_value = [("id-1", "USER enjoys hiking", 0.95)]
    assert rc.find_memory_contradiction(mem, new) is None

    mem.search.return_value = [
        ("id-1", "USER enjoys hiking", 0.95),
        ("id-2", existing, 0.92),
    ]
    hit = rc.find_memory_contradiction(mem, new)
    assert hit == ("id-2", existing, 0.92)


@pytest.mark.unit
def test_find_diary_contradiction_uses_search_with_ids(monkeypatch):
    monkeypatch.setattr(rc.settings, "memory_similarity_floor", 0.3)
    diary = MagicMock()
    new = "I do not feel confident when debugging async code"
    existing = "I feel confident when debugging async code"
    diary.search_with_ids.return_value = [("rid-1", existing, 0.91, None)]
    hit = rc.find_diary_contradiction(diary, new)
    assert hit == ("rid-1", existing, 0.91)
    diary.search_with_ids.assert_called_once()


@pytest.mark.unit
def test_apply_memory_contradiction_policy_log(monkeypatch):
    monkeypatch.setattr(rc.settings, "memory_contradiction_policy", "log")
    monkeypatch.setattr(rc.settings, "memory_contradiction_importance_factor", 0.7)
    mem = MagicMock()
    mem.save.return_value = "new-id"
    mem.scale_importance.return_value = True
    events: list[tuple[str, dict]] = []
    outcomes: list[str] = []

    def capture(event: str, **fields):
        events.append((event, fields))

    monkeypatch.setattr(rc, "log_event", capture)
    monkeypatch.setattr(
        "src.policy_knobs.apply_outcome_safe",
        lambda signal, **kw: outcomes.append(signal),
    )
    ok = rc.apply_memory_contradiction(
        mem,
        "USER does not use Python",
        ("old-id", "USER uses Python", 0.9),
    )
    assert ok is True
    assert events[0][0] == "reflect.contradiction_candidate"
    assert events[0][1]["policy"] == "log"
    mem.scale_importance.assert_called_once_with(
        "old-id", 0.7, agent=rc.settings.default_agent_id
    )
    assert outcomes == ["log_pile"]
    mem.save.assert_called_once_with(
        "USER does not use Python",
        memory_type="semantic",
        agent=rc.settings.default_agent_id,
    )


@pytest.mark.unit
def test_log_pile_nudge_survives_metrics_failure(monkeypatch):
    monkeypatch.setattr(rc.settings, "memory_contradiction_policy", "log")
    outcomes: list[str] = []
    monkeypatch.setattr(
        "src.obs.metrics.record_memory_contradiction_event",
        lambda **kw: (_ for _ in ()).throw(RuntimeError("otel down")),
    )
    monkeypatch.setattr(
        "src.policy_knobs.apply_outcome_safe",
        lambda signal, **kw: outcomes.append(signal),
    )
    rc.log_contradiction_candidate(
        lane="memories",
        new_text="USER does not use Python",
        existing_id="old-id",
        existing_text="USER uses Python",
        similarity=0.9,
        policy="log",
    )
    assert outcomes == ["log_pile"]


@pytest.mark.unit
def test_diary_log_policy_does_not_nudge_retrieve_knobs(monkeypatch):
    outcomes: list[str] = []
    monkeypatch.setattr(
        "src.policy_knobs.apply_outcome_safe",
        lambda signal, **kw: outcomes.append(signal),
    )
    monkeypatch.setattr(rc, "log_event", lambda *a, **k: None)
    rc.log_contradiction_candidate(
        lane="self_reflections",
        new_text="I do not feel confident",
        existing_id="rid-1",
        existing_text="I feel confident",
        similarity=0.9,
        policy="log",
    )
    assert outcomes == []


@pytest.mark.unit
def test_apply_memory_contradiction_policy_unverified(monkeypatch):
    monkeypatch.setattr(rc.settings, "memory_contradiction_policy", "unverified")
    mem = MagicMock()
    mem.save.return_value = "new-id"
    monkeypatch.setattr(rc, "log_event", lambda *a, **k: None)
    ok = rc.apply_memory_contradiction(
        mem,
        "USER does not use Python",
        ("old-id", "USER uses Python", 0.9),
    )
    assert ok is True
    mem.save.assert_called_once_with(
        "USER does not use Python",
        memory_type="unverified",
        agent=rc.settings.default_agent_id,
    )


@pytest.mark.unit
def test_apply_memory_contradiction_policy_replace(monkeypatch):
    monkeypatch.setattr(rc.settings, "memory_contradiction_policy", "replace")
    mem = MagicMock()
    mem.replace.return_value = True
    events: list[str] = []
    monkeypatch.setattr(
        rc,
        "log_event",
        lambda event, **fields: events.append(event),
    )
    ok = rc.apply_memory_contradiction(
        mem,
        "USER does not use Python",
        ("old-id", "USER uses Python", 0.9),
    )
    assert ok is True
    mem.replace.assert_called_once_with(
        "old-id",
        "USER does not use Python",
        agent=rc.settings.default_agent_id,
    )
    assert "reflect.contradiction_resolved" in events


@pytest.mark.unit
def test_apply_diary_contradiction_policy_unverified_skips(monkeypatch):
    monkeypatch.setattr(rc.settings, "memory_contradiction_policy", "unverified")
    diary = MagicMock()
    events: list[str] = []
    monkeypatch.setattr(
        rc,
        "log_event",
        lambda event, **fields: events.append(event),
    )
    ok = rc.apply_diary_contradiction(
        diary,
        "I do not feel calm",
        ("rid", "I feel calm", 0.9),
    )
    assert ok is False
    diary.save.assert_not_called()
    assert "reflect.contradiction_skipped" in events


@pytest.mark.unit
def test_reflect_contradiction_candidate_visible_in_logs(
    caplog: pytest.LogCaptureFixture,
):
    setup_agent_event_logger()
    log_event(
        "reflect.contradiction_candidate",
        lane="memories",
        policy="log",
        similarity=0.91,
        similarity_threshold=0.86,
        existing_id="abc",
        new_preview="USER does not use Python",
        existing_preview="USER uses Python",
    )
    assert any(
        '"event": "reflect.contradiction_candidate"' in r.message
        for r in caplog.records
    )
