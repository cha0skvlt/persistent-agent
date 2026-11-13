"""Diary echo gate — deferred reflect near-duplicate skip."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.graph import diary_echo_gate as deg
from src.memory import PROTECTED_DIARY_EXACT


def _unit_vec(dim: int, index: int) -> list[float]:
    v = [0.0] * dim
    v[index] = 1.0
    return v


@pytest.mark.unit
def test_find_diary_echo_match_above_threshold():
    diary = MagicMock()
    diary.get_vector.return_value = _unit_vec(4, 0)
    diary.recent_rows_with_embeddings.return_value = [
        ("dup-id", "you give light, I work it off", _unit_vec(4, 0)),
    ]

    match = deg.find_diary_echo_match(
        diary, "you give light — I work it off", threshold=0.92
    )
    assert match is not None
    assert match[0] == "dup-id"
    assert match[2] == pytest.approx(1.0)


@pytest.mark.unit
def test_find_diary_echo_match_below_threshold():
    diary = MagicMock()
    diary.get_vector.return_value = _unit_vec(4, 0)
    diary.recent_rows_with_embeddings.return_value = [
        ("other-id", "a completely different thought", _unit_vec(4, 1)),
    ]

    assert deg.find_diary_echo_match(diary, "a new thought", threshold=0.92) is None


@pytest.mark.unit
@pytest.mark.parametrize("seed", sorted(PROTECTED_DIARY_EXACT))
def test_protected_seed_exempt_from_echo_check(seed: str):
    diary = MagicMock()
    assert deg.is_diary_echo_exempt(seed) is True
    assert deg.find_diary_echo_match(diary, seed) is None
    diary.get_vector.assert_not_called()


@pytest.mark.unit
def test_skip_diary_echo_duplicate_logs_event(monkeypatch):
    captured: list[tuple[str, dict]] = []

    def capture(event: str, **fields):
        captured.append((event, fields))

    monkeypatch.setattr(deg, "log_event", capture)
    # Novelty fail-open / novel → cosine lookback still applies.
    monkeypatch.setattr(
        "src.pulse.novelty.is_novel", lambda *_a, **_k: True, raising=False
    )
    diary = MagicMock()
    diary.get_vector.return_value = [1.0, 0.0]
    diary.recent_rows_with_embeddings.return_value = [
        ("near-id", "echo line", [1.0, 0.0]),
    ]

    assert deg.skip_diary_echo_duplicate(diary, "echo line again") is True
    assert captured[0][0] == "reflect.diary_echo_skipped"
    assert captured[0][1]["nearest_id"] == "near-id"


@pytest.mark.unit
def test_skip_diary_echo_via_novelty(monkeypatch):
    captured: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        deg, "log_event", lambda event, **fields: captured.append((event, fields))
    )
    monkeypatch.setattr(
        "src.pulse.novelty.is_novel", lambda *_a, **_k: False, raising=False
    )
    diary = MagicMock()
    assert deg.skip_diary_echo_duplicate(diary, "same as recent diary") is True
    assert captured[0][1].get("via") == "novelty"
    diary.get_vector.assert_not_called()
