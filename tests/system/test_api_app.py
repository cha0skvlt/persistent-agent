"""FastAPI app tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from src.agent_runner import AgentTurnResult
from src.config import settings


@pytest.fixture
def api_client(monkeypatch):
    monkeypatch.setattr(settings, "telegram_bot_token", "", raising=False)
    monkeypatch.setattr(settings, "telegram_polling", False, raising=False)
    monkeypatch.setattr(settings, "api_key", "", raising=False)
    with (
        patch("src.api.app.verify_boot_dependencies_with_retry"),
        patch("src.api.app.get_graph_app", return_value=object()),
    ):
        from src.api.app import app

        with TestClient(app) as client:
            yield client


@pytest.mark.unit
@pytest.mark.db
def test_health(api_client):
    resp = api_client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["telegram_polling"] is False


@pytest.mark.unit
def test_chat(api_client, monkeypatch):
    monkeypatch.setattr(
        "src.api.app.arun_agent_turn",
        AsyncMock(
            return_value=AgentTurnResult(
                reply="AGENT reply",
                tools_used=(),
                elapsed_ms=100.0,
            )
        ),
    )
    resp = api_client.post("/chat", json={"message": "hello"})
    assert resp.status_code == 200
    assert resp.json()["reply"] == "AGENT reply"


@pytest.mark.unit
def test_chat_schedules_deferred_summary(api_client, monkeypatch):
    from unittest.mock import MagicMock

    scheduled: list[str | None] = []

    def _track(coro, *, name=None):
        scheduled.append(name)
        coro.close()
        return MagicMock()

    monkeypatch.setattr("src.api.app.asyncio.create_task", _track)
    monkeypatch.setattr(
        "src.api.app.arun_agent_turn",
        AsyncMock(
            return_value=AgentTurnResult(
                reply="ok",
                tools_used=(),
                elapsed_ms=1.0,
                summary_pending=True,
                thread_id="api-thread",
            )
        ),
    )
    resp = api_client.post("/chat", json={"message": "hello"})
    assert resp.status_code == 200
    assert scheduled


@pytest.mark.unit
def test_chat_requires_api_key(api_client, monkeypatch):
    monkeypatch.setattr(settings, "api_key", "secret", raising=False)
    monkeypatch.setattr(
        "src.api.app.arun_agent_turn",
        AsyncMock(
            return_value=AgentTurnResult(reply="ok", tools_used=(), elapsed_ms=1.0)
        ),
    )
    resp = api_client.post("/chat", json={"message": "hello"})
    assert resp.status_code == 401
    resp2 = api_client.post(
        "/chat",
        json={"message": "hello"},
        headers={"Authorization": "Bearer secret"},
    )
    assert resp2.status_code == 200


@pytest.mark.unit
def test_chat_empty_thread_id(api_client, monkeypatch):
    monkeypatch.setattr(settings, "api_default_thread_id", "  ", raising=False)
    resp = api_client.post("/chat", json={"message": "hello"})
    assert resp.status_code == 400
