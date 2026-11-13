"""Deferred job overlap tracker (instrumentation only)."""

from __future__ import annotations

import pytest

from src.obs import deferred_job_tracker as djt


@pytest.mark.unit
def test_track_deferred_job_marks_active():
    djt.reset_deferred_job_tracker()
    assert not djt.deferred_job_active("telegram_1")
    with djt.track_deferred_job("telegram_1", "reflect"):
        assert djt.deferred_job_active("telegram_1")
        assert djt.active_deferred_jobs("telegram_1") == frozenset({"reflect"})
    assert not djt.deferred_job_active("telegram_1")


@pytest.mark.unit
def test_live_turn_overlaps_deferred_before_lock():
    djt.reset_deferred_job_tracker()
    with djt.track_deferred_job("telegram_5", "summary"):
        assert djt.live_turn_overlaps_deferred("telegram_5") is True
        assert djt.live_turn_overlaps_deferred("telegram_9") is False


@pytest.mark.unit
def test_traced_agent_turn_sets_overlap_attribute():
    from unittest.mock import MagicMock, patch

    from src.obs.spans import traced_agent_turn

    mock_tracer = MagicMock()
    mock_span = MagicMock()
    mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
    with (
        patch("src.obs.spans.is_telemetry_enabled", return_value=True),
        patch("src.obs.spans.tracer", mock_tracer),
    ):
        with traced_agent_turn(
            thread_id="telegram_1",
            channel="telegram",
            concurrent_deferred_job=True,
        ):
            pass
    mock_span.set_attribute.assert_any_call("agent.concurrent_deferred_job", True)


@pytest.mark.unit
def test_record_turn_duration_includes_overlap_label(monkeypatch):
    from unittest.mock import MagicMock

    from src.obs import metrics

    hist = MagicMock()
    monkeypatch.setattr(metrics, "turn_duration", hist)
    monkeypatch.setattr(metrics, "is_telemetry_enabled", lambda: True)
    metrics.record_turn_duration(2.5, "telegram", concurrent_deferred_job=True)
    hist.record.assert_called_once_with(
        2.5,
        {"channel": "telegram", "concurrent_deferred_job": "true"},
    )
