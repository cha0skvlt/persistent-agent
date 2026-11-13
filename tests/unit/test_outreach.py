"""Unit tests for proactive outreach."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.config import settings
from src.heart import outreach as outreach_mod


class _Resp:
    def __init__(self, ok: bool, status: int = 200, description: str = ""):
        self.status_code = status
        self._ok = ok
        self.text = description
        self._description = description

    def json(self):
        data: dict = {"ok": self._ok}
        if self._description:
            data["description"] = self._description
        return data


@pytest.mark.unit
def test_send_telegram_strips_think_for_outreach(monkeypatch):
    """Proactive send uses outreach=True — <think> never reaches Telegram."""
    monkeypatch.setattr(settings, "telegram_bot_token", "tok", raising=False)
    monkeypatch.setattr(settings, "telegram_owner_chat_id", 123, raising=False)
    calls: list[dict] = []

    def _post(url, json=None, timeout=10.0):
        calls.append(json or {})
        return _Resp(True)

    monkeypatch.setattr("src.api.telegram_send.httpx.post", _post)
    ok = outreach_mod._send_telegram(
        "<think>my private reasoning</think>\n\nHello USER",
        "**→ turn 2.1s**  search_diary",
    )
    assert ok is True
    html = calls[0]["text"]
    assert calls[0]["parse_mode"] == "HTML"
    assert "<blockquote" not in html
    assert "my private reasoning" not in html
    assert "Hello USER" in html
    assert "<b>→ turn 2.1s</b>" in html
    assert "search_diary" in html


@pytest.mark.unit
def test_send_telegram_html_then_plain_fallback(monkeypatch):
    monkeypatch.setattr(settings, "telegram_bot_token", "tok", raising=False)
    monkeypatch.setattr(settings, "telegram_owner_chat_id", 123, raising=False)
    calls: list[dict] = []

    def _post(url, json=None, timeout=10.0):
        calls.append(json or {})
        if len(calls) == 1:
            return _Resp(False, status=400, description="can't parse entities")
        return _Resp(True)

    monkeypatch.setattr("src.api.telegram_send.httpx.post", _post)
    assert outreach_mod._send_telegram("**bold** and text") is True
    assert calls[0].get("parse_mode") == "HTML"
    assert "parse_mode" not in calls[1]


@pytest.mark.unit
def test_send_telegram_plain_fallback_checks_ok(monkeypatch):
    monkeypatch.setattr(settings, "telegram_bot_token", "tok", raising=False)
    monkeypatch.setattr(settings, "telegram_owner_chat_id", 123, raising=False)

    def _post(url, json=None, timeout=10.0):
        if json and json.get("parse_mode") == "HTML":
            return _Resp(False, status=400, description="can't parse entities")
        return _Resp(False, status=200)

    monkeypatch.setattr("src.api.telegram_send.httpx.post", _post)
    assert outreach_mod._send_telegram("hello") is False


@pytest.mark.unit
def test_send_telegram_no_fallback_on_non_parse_error(monkeypatch):
    """A generic failure (not parse/too-long) must NOT retry as plain."""
    monkeypatch.setattr(settings, "telegram_bot_token", "tok", raising=False)
    monkeypatch.setattr(settings, "telegram_owner_chat_id", 123, raising=False)
    calls: list[dict] = []

    def _post(url, json=None, timeout=10.0):
        calls.append(json or {})
        return _Resp(False, status=403, description="bot was blocked by the user")

    monkeypatch.setattr("src.api.telegram_send.httpx.post", _post)
    assert outreach_mod._send_telegram("hello") is False
    assert len(calls) == 1


@pytest.mark.unit
def test_send_outreach_kill_switch(monkeypatch, reset_outreach_state):
    monkeypatch.setattr(settings, "outreach_enabled", False, raising=False)
    with patch("src.api.telegram_send.httpx.post") as post:
        outreach_mod.send_outreach_if_due(MagicMock())
        post.assert_not_called()


@pytest.mark.unit
def test_send_outreach_skips_when_asleep(monkeypatch, reset_outreach_state):
    monkeypatch.setattr(settings, "outreach_enabled", True, raising=False)
    monkeypatch.setattr(settings, "telegram_owner_chat_id", 1, raising=False)
    monkeypatch.setattr(settings, "telegram_bot_token", "tok", raising=False)
    monkeypatch.setattr("src.agent_mode.is_asleep", lambda: True)
    conn = MagicMock()
    with patch.object(outreach_mod, "_send_telegram") as send:
        outreach_mod.send_outreach_if_due(conn)
        send.assert_not_called()
    conn.execute.assert_not_called()


@pytest.mark.unit
def test_send_outreach_skips_without_bot_token(monkeypatch, reset_outreach_state):
    monkeypatch.setattr(settings, "outreach_enabled", True, raising=False)
    monkeypatch.setattr(settings, "telegram_owner_chat_id", "123", raising=False)
    monkeypatch.setattr(settings, "telegram_bot_token", "", raising=False)
    conn = MagicMock()
    with patch.object(outreach_mod, "_send_telegram") as send:
        outreach_mod.send_outreach_if_due(conn)
        send.assert_not_called()
    assert outreach_mod._last_send_fail_at == 0.0


@pytest.mark.unit
def test_send_outreach_happy_path(monkeypatch, reset_outreach_state):
    monkeypatch.setattr(settings, "outreach_enabled", True, raising=False)
    monkeypatch.setattr(settings, "telegram_owner_chat_id", 1, raising=False)
    monkeypatch.setattr(settings, "telegram_bot_token", "tok", raising=False)
    outbox_id = "00000000-0000-0000-0000-000000000001"
    conn = MagicMock()
    recover_cur = MagicMock()
    recover_cur.fetchall.return_value = []
    claim_cur = MagicMock()
    claim_cur.fetchone.return_value = (
        outbox_id,
        "Hello user",
        "**→ turn 1.2s**  search_diary",
        1200.0,
    )
    sent_cur = MagicMock()
    conn.execute.side_effect = [recover_cur, claim_cur, sent_cur]
    sent_args: list[tuple] = []

    def _send(text, footer=""):
        sent_args.append((text, footer))
        return True

    with (
        patch.object(outreach_mod, "_send_telegram", side_effect=_send),
        patch.object(outreach_mod, "_save_diary", return_value=True),
        patch("src.mind.hub.record_outreach_sent"),
    ):
        outreach_mod.send_outreach_if_due(conn)
    claim_sql = conn.execute.call_args_list[1].args[0]
    assert "FOR UPDATE SKIP LOCKED" in claim_sql
    assert "status = 'sending'" in claim_sql
    assert "footer_text" in claim_sql
    assert "make_interval" in claim_sql
    assert sent_args == [("Hello user", "**→ turn 1.2s**  search_diary")]
    update_calls = [
        c
        for c in conn.execute.call_args_list
        if c.args and "UPDATE outbox SET status = 'sent'" in str(c.args[0])
    ]
    assert update_calls
    assert outbox_id in update_calls[0].args[1]


@pytest.mark.unit
def test_send_outreach_empty_footer_gets_synthetic(monkeypatch, reset_outreach_state):
    """Past grace / orphan row — duration from queue age, always includes message_user."""
    monkeypatch.setattr(settings, "outreach_enabled", True, raising=False)
    monkeypatch.setattr(settings, "telegram_owner_chat_id", 1, raising=False)
    monkeypatch.setattr(settings, "telegram_bot_token", "tok", raising=False)
    outbox_id = "00000000-0000-0000-0000-000000000099"
    conn = MagicMock()
    recover_cur = MagicMock()
    recover_cur.fetchall.return_value = []
    claim_cur = MagicMock()
    # age_ms ≈ 95s past grace
    claim_cur.fetchone.return_value = (outbox_id, "Hello without stamp", "", 95_000.0)
    sent_cur = MagicMock()
    conn.execute.side_effect = [recover_cur, claim_cur, sent_cur]
    sent_args: list[tuple] = []

    def _send(text, footer=""):
        sent_args.append((text, footer))
        return True

    with (
        patch.object(outreach_mod, "_send_telegram", side_effect=_send),
        patch.object(outreach_mod, "_save_diary", return_value=True),
        patch("src.mind.hub.record_outreach_sent"),
    ):
        outreach_mod.send_outreach_if_due(conn)
    assert sent_args == [("Hello without stamp", "**→ turn 1m 35s**  message_user")]


@pytest.mark.unit
def test_send_outreach_upgrades_blank_duration_footer(
    monkeypatch, reset_outreach_state
):
    """Stamped ``→ turn —  message_user`` still gets a real duration at send."""
    monkeypatch.setattr(settings, "outreach_enabled", True, raising=False)
    monkeypatch.setattr(settings, "telegram_owner_chat_id", 1, raising=False)
    monkeypatch.setattr(settings, "telegram_bot_token", "tok", raising=False)
    outbox_id = "00000000-0000-0000-0000-000000000098"
    conn = MagicMock()
    recover_cur = MagicMock()
    recover_cur.fetchall.return_value = []
    claim_cur = MagicMock()
    claim_cur.fetchone.return_value = (
        outbox_id,
        "Hello",
        "**→ turn —**  message_user",
        120_000.0,
    )
    sent_cur = MagicMock()
    conn.execute.side_effect = [recover_cur, claim_cur, sent_cur]
    sent_args: list[tuple] = []

    with (
        patch.object(
            outreach_mod,
            "_send_telegram",
            side_effect=lambda text, footer="": sent_args.append((text, footer))
            or True,
        ),
        patch.object(outreach_mod, "_save_diary", return_value=True),
        patch("src.mind.hub.record_outreach_sent"),
    ):
        outreach_mod.send_outreach_if_due(conn)
    assert sent_args == [("Hello", "**→ turn 2m 0s**  message_user")]


@pytest.mark.unit
def test_send_outreach_drops_operator_error_draft(monkeypatch, reset_outreach_state):
    monkeypatch.setattr(settings, "outreach_enabled", True, raising=False)
    monkeypatch.setattr(settings, "telegram_owner_chat_id", 1, raising=False)
    monkeypatch.setattr(settings, "telegram_bot_token", "tok", raising=False)
    outbox_id = "00000000-0000-0000-0000-000000000077"
    poison = (
        "Unable to generate parser for this template. "
        "No user query found in messages. invalid_request_error"
    )
    conn = MagicMock()
    recover_cur = MagicMock()
    recover_cur.fetchall.return_value = []
    claim_cur = MagicMock()
    claim_cur.fetchone.return_value = (outbox_id, poison, "**→ turn 9m**", 9_000.0)
    drop_cur = MagicMock()
    conn.execute.side_effect = [recover_cur, claim_cur, drop_cur]
    with patch.object(outreach_mod, "_send_telegram") as send:
        outreach_mod.send_outreach_if_due(conn)
        send.assert_not_called()
    assert "status = 'sent'" in str(conn.execute.call_args_list[2].args[0])


@pytest.mark.unit
def test_claim_skips_young_rows_without_footer(monkeypatch, reset_outreach_state):
    """Heart must wait for post-graph stamp (grace window)."""
    monkeypatch.setattr(settings, "outreach_enabled", True, raising=False)
    monkeypatch.setattr(settings, "telegram_owner_chat_id", 1, raising=False)
    monkeypatch.setattr(settings, "telegram_bot_token", "tok", raising=False)
    monkeypatch.setattr(settings, "outreach_footer_grace_sec", 90.0, raising=False)
    conn = MagicMock()
    recover_cur = MagicMock()
    recover_cur.fetchall.return_value = []
    claim_cur = MagicMock()
    claim_cur.fetchone.return_value = None
    conn.execute.side_effect = [recover_cur, claim_cur]
    with patch.object(outreach_mod, "_send_telegram") as send:
        outreach_mod.send_outreach_if_due(conn)
        send.assert_not_called()
    claim_sql = conn.execute.call_args_list[1].args[0]
    assert "footer_text" in claim_sql
    assert "make_interval" in claim_sql
    assert conn.execute.call_args_list[1].args[1][1] == 90.0


@pytest.mark.unit
def test_send_outreach_telegram_fail_releases_claim(monkeypatch, reset_outreach_state):
    monkeypatch.setattr(settings, "outreach_enabled", True, raising=False)
    monkeypatch.setattr(settings, "telegram_owner_chat_id", 1, raising=False)
    monkeypatch.setattr(settings, "telegram_bot_token", "tok", raising=False)
    outbox_id = "00000000-0000-0000-0000-000000000002"
    conn = MagicMock()
    recover_cur = MagicMock()
    recover_cur.fetchall.return_value = []
    claim_cur = MagicMock()
    claim_cur.fetchone.return_value = (outbox_id, "draft", "**→ turn 1.0s**", 1000.0)
    release_cur = MagicMock()
    release_cur.fetchone.return_value = (1, "pending")
    conn.execute.side_effect = [recover_cur, claim_cur, release_cur]
    with (
        patch.object(outreach_mod, "_send_telegram", return_value=False),
        patch.object(outreach_mod, "_save_diary") as diary,
    ):
        outreach_mod.send_outreach_if_due(conn)
    diary.assert_not_called()
    release_sql = conn.execute.call_args_list[2].args[0]
    assert "attempts = attempts + 1" in release_sql
    assert "next_attempt_at" in release_sql
    assert "'dead'" in release_sql
    assert "status = 'sending'" in release_sql
    assert not any(
        "UPDATE outbox SET status = 'sent'" in str(c.args[0])
        for c in conn.execute.call_args_list
    )


@pytest.mark.unit
def test_send_outreach_marks_sent_before_diary(monkeypatch, reset_outreach_state):
    """Crash after Telegram must not leave row pending — mark sent first."""
    monkeypatch.setattr(settings, "outreach_enabled", True, raising=False)
    monkeypatch.setattr(settings, "telegram_owner_chat_id", 1, raising=False)
    monkeypatch.setattr(settings, "telegram_bot_token", "tok", raising=False)
    outbox_id = "00000000-0000-0000-0000-000000000003"
    order: list[str] = []
    conn = MagicMock()
    recover_cur = MagicMock()
    recover_cur.fetchall.return_value = []
    claim_cur = MagicMock()
    claim_cur.fetchone.return_value = (outbox_id, "Hi", "**→ turn 0.5s**", 500.0)
    sent_cur = MagicMock()
    conn.execute.side_effect = [recover_cur, claim_cur, sent_cur]

    def _send(_text, _footer=""):
        order.append("send")
        return True

    def _diary(_conn, _draft):
        order.append("diary")
        return True

    with (
        patch.object(outreach_mod, "_send_telegram", side_effect=_send),
        patch.object(outreach_mod, "_save_diary", side_effect=_diary),
        patch(
            "src.mind.hub.record_outreach_sent",
            side_effect=lambda **_: order.append("hub"),
        ),
    ):
        outreach_mod.send_outreach_if_due(conn)
    assert order == ["send", "diary", "hub"]
    # Third execute is mark-sent (after claim); happens before diary returns.
    assert "status = 'sent'" in str(conn.execute.call_args_list[2].args[0])
