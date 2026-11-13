"""Integration tests for outreach pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.config import settings
from src.heart import outreach as outreach_mod
from src.heart.heartbeat import tick


@pytest.mark.integration
def test_send_outreach_claims_oldest_pending(monkeypatch, reset_outreach_state):
    monkeypatch.setattr(settings, "outreach_enabled", True, raising=False)
    monkeypatch.setattr(settings, "telegram_owner_chat_id", "12345", raising=False)
    monkeypatch.setattr(settings, "telegram_bot_token", "tok", raising=False)
    conn = MagicMock()
    recover_cur = MagicMock()
    recover_cur.fetchall.return_value = []
    claim_cur = MagicMock()
    claim_cur.fetchone.return_value = (
        "00000000-0000-0000-0000-000000000099",
        "oldest draft",
        "**→ turn 1s**",
        1000.0,
    )
    sent_cur = MagicMock()
    conn.execute.side_effect = [recover_cur, claim_cur, sent_cur]
    with (
        patch.object(outreach_mod, "_send_telegram", return_value=True),
        patch.object(outreach_mod, "_save_diary", return_value=True),
        patch("src.mind.hub.record_outreach_sent"),
    ):
        outreach_mod.send_outreach_if_due(conn)
    claim_sql = conn.execute.call_args_list[1].args[0]
    assert "FOR UPDATE SKIP LOCKED" in claim_sql
    assert "ORDER BY created_at" in claim_sql
    assert "status = 'sending'" in claim_sql


@pytest.mark.integration
def test_heartbeat_tick_invokes_outreach(monkeypatch):
    monkeypatch.setattr(
        "src.heart.heartbeat.collect_body_metrics",
        lambda: {"disk_usage_pct": 1.0},
    )
    monkeypatch.setattr(
        "src.heart.heartbeat._compute_phylactery_status",
        lambda: ("ok", {}),
    )
    monkeypatch.setattr("src.heart.heartbeat.latest_tick", lambda _c: None)
    monkeypatch.setattr("src.heart.heartbeat._maybe_schedule_backup", lambda _p: None)
    monkeypatch.setattr("src.heart.heartbeat.write_tick", lambda *a, **k: 1)
    monkeypatch.setattr("src.heart.heartbeat.sign_tick_payload", lambda *_a: None)

    conn = MagicMock()
    with patch("src.heart.heartbeat.send_outreach_if_due") as send:
        tick(conn)
        send.assert_called_once_with(conn)
