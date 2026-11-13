"""Agent sleep/wake mode — file-backed flag (no Ollama I/O in unit tests)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.agent_mode import (
    MODE_AWAKE,
    MODE_SLEEP,
    get_mode,
    is_asleep,
    set_mode,
)


@pytest.fixture()
def mode_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "agent_mode.json"
    monkeypatch.setattr(
        "src.agent_mode.settings.agent_mode_path",
        str(path),
    )
    return path


def test_default_awake_when_missing(mode_file: Path):
    assert not mode_file.exists()
    assert get_mode() == MODE_AWAKE
    assert is_asleep() is False


def test_set_and_get_sleep_wake(mode_file: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("src.agent_mode.log_event", lambda *a, **k: None)
    assert set_mode(MODE_SLEEP, source="test") == MODE_SLEEP
    assert is_asleep() is True
    assert mode_file.exists()
    assert set_mode(MODE_AWAKE, source="test") == MODE_AWAKE
    assert is_asleep() is False


def test_corrupt_file_fails_open_awake(mode_file: Path):
    mode_file.write_text("{not-json", encoding="utf-8")
    assert get_mode() == MODE_AWAKE


def test_invalid_mode_raises(mode_file: Path):
    with pytest.raises(ValueError, match="invalid agent mode"):
        set_mode("hibernate")


def test_unknown_mode_value_in_file_fails_open(mode_file: Path):
    mode_file.write_text('{"mode": "nap"}', encoding="utf-8")
    assert get_mode() == MODE_AWAKE
