"""Loop budget: fruitless detection must not fire on useful content."""

from __future__ import annotations

import pytest

from src.graph.loop_budget import round_is_fruitless


@pytest.mark.unit
def test_empty_round_is_fruitless():
    assert round_is_fruitless([]) is True
    assert round_is_fruitless(["", "   "]) is True


@pytest.mark.unit
def test_structured_failures_are_fruitless():
    for text in (
        "browse_error: no session",
        "lesson_error: save_failed",
        "lesson_empty: no matching lessons",
        "deixis_error: use USER: prefix",
        "list_failed: timeout",
        "cancel_failed: timeout",
        "distill_rejected:no_evidence",
        "Not scheduled: text is empty.",
        "Not sent: queue full",
        "not_found",
        "empty_content",
        "no valid ids",
        "No results for: query",
        "Web search unavailable: timeout",
        "HTTP 404 from host",
        "[]",
    ):
        assert round_is_fruitless([text]) is True, text


@pytest.mark.unit
def test_useful_content_containing_failure_words_is_not_fruitless():
    # The regression: substring matching stopped the loop on good material.
    for text in (
        "Error handling in PostgreSQL uses savepoints and retry loops.",
        "The empty table case is covered by the migration.",
        "Failed logins are rate limited after five attempts.",
        "A list of errors and how to recover from each one.",
    ):
        assert round_is_fruitless([text]) is False, text


@pytest.mark.unit
def test_one_useful_result_keeps_the_round_alive():
    assert (
        round_is_fruitless(["not_found", "Postgres uses MVCC for isolation"]) is False
    )
