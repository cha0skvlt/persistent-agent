"""Background tool allowlist — intent may use run_selftest."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.graph.workflow import _tools_hidden_on_turn
from src.reverie.turn_kind import TurnKind, turn_kind


@pytest.mark.unit
def test_live_turn_hides_message_user():
    with patch("src.reverie.background.is_reverie_background_pass", return_value=False):
        hidden = _tools_hidden_on_turn({"diary_source": ""})
    assert "message_user" in hidden
    assert "run_selftest" not in hidden


@pytest.mark.unit
def test_solitary_binds_full_tool_belt():
    with patch("src.reverie.background.is_reverie_background_pass", return_value=True):
        hidden = _tools_hidden_on_turn({"diary_source": "solitary"})
    assert hidden == frozenset()
    assert "run_selftest" not in hidden
    assert "message_user" not in hidden


@pytest.mark.unit
def test_intent_allows_selftest():
    with patch("src.reverie.background.is_reverie_background_pass", return_value=True):
        hidden = _tools_hidden_on_turn({"diary_source": "intent"})
    assert "run_selftest" not in hidden
    assert "affirm_identity" not in hidden


@pytest.mark.unit
def test_intent_turnkind_alone_allows_selftest():
    """TurnKind.INTENT must unlock selftest even when diary_source is empty."""
    with turn_kind(TurnKind.INTENT):
        hidden = _tools_hidden_on_turn({"diary_source": ""})
    assert "run_selftest" not in hidden
    assert "message_user" not in hidden


@pytest.mark.unit
def test_intent_diary_source_alone_is_background():
    """diary_source=intent without TurnKind still gets background tool set."""
    with patch("src.reverie.background.is_reverie_background_pass", return_value=False):
        hidden = _tools_hidden_on_turn({"diary_source": "intent"})
    assert "run_selftest" not in hidden
    assert "message_user" not in hidden
