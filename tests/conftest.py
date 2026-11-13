"""Root pytest config — shared fixtures and path-based markers (unit / integration / system)."""

from __future__ import annotations

import os

import pytest

import src.graph.checkpoint_serde  # noqa: F401 — Reviver allowlist before langgraph
from src.app_runtime import reset_process_state_for_tests
from src.config import settings
from src.db.pool import close_pool


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "unit: pure logic, mocked IO")
    config.addinivalue_line(
        "markers", "integration: multi-module / graph behavior, mocked backends"
    )
    config.addinivalue_line("markers", "system: API, heart, live stack, import smoke")
    config.addinivalue_line(
        "markers", "db: requires TEST_DATABASE_URL (destructive TRUNCATE)"
    )
    config.addinivalue_line(
        "markers", "live: requires reachable Ollama (REASONING_MODEL / embed smoke)"
    )


def _path_tier(nodeid: str) -> str | None:
    path = nodeid.replace("\\", "/")
    for tier in ("unit", "integration", "system"):
        if f"/tests/{tier}/" in path or path.startswith(f"tests/{tier}/"):
            return tier
    return None


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    for item in items:
        tier = _path_tier(item.nodeid)
        if tier == "unit":
            item.add_marker(pytest.mark.unit)
        elif tier == "integration":
            item.add_marker(pytest.mark.integration)
        elif tier == "system":
            item.add_marker(pytest.mark.system)

    # DB-marked tests need a disposable Postgres; skip instead of failing on a timeout.
    if not (os.environ.get("TEST_DATABASE_URL") or "").strip():
        skip_db = pytest.mark.skip(reason="TEST_DATABASE_URL not set (DB-marked test)")
        for item in items:
            if "db" in item.keywords:
                item.add_marker(skip_db)


@pytest.fixture(autouse=True)
def _disable_langchain_cloud_tracing(monkeypatch: pytest.MonkeyPatch):
    """Keep LangChain cloud tracing env off during tests."""
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)


@pytest.fixture(autouse=True)
def _reset_agent_event_logger():
    """Prevent OTEL handlers on ``agent.event`` from breaking caplog in later tests."""
    import logging

    log = logging.getLogger("agent.event")
    log.handlers.clear()
    log.propagate = True
    yield
    log.handlers.clear()
    log.propagate = True


@pytest.fixture(autouse=True)
def _telegram_allow_all_chats(monkeypatch: pytest.MonkeyPatch):
    """Tests use explicit allowlist — production is fail-closed when empty."""
    monkeypatch.setattr(
        settings, "telegram_allowed_chat_ids", [1, 5, 7, 99], raising=False
    )
    monkeypatch.setattr(settings, "telegram_owner_chat_id", "", raising=False)


@pytest.fixture(autouse=True)
def _disable_otel_export(monkeypatch: pytest.MonkeyPatch):
    """Keep unit tests free of OTLP exporters unless a test opts in."""
    monkeypatch.setattr(settings, "otel_enabled", False, raising=False)


@pytest.fixture(autouse=True)
def _disable_web_render_fallback(monkeypatch: pytest.MonkeyPatch):
    """Static-fetch tests must not launch a real Chromium render (opt in per test)."""
    monkeypatch.setattr(settings, "web_fetch_render_fallback", False, raising=False)


@pytest.fixture(autouse=True)
def _fast_db_fail(monkeypatch: pytest.MonkeyPatch):
    """No test may wait 30s on a missing Postgres.

    A real pool wait costs ``DB_POOL_TIMEOUT`` per call. A handful of tests reached
    the DB by accident and the suite ran 20 minutes. Fail in 0.1s instead; callers
    that tolerate a DB error (``PolicyKnobs.get_all``) fall back to defaults.
    """
    monkeypatch.setattr(settings, "db_pool_timeout", 0.1, raising=False)
    close_pool()
    yield
    close_pool()


@pytest.fixture(autouse=True)
def _reset_app_runtime():
    """Single process wipe — all singleton resets go through ``app_runtime``."""
    reset_process_state_for_tests()
    yield
    reset_process_state_for_tests()


@pytest.fixture(autouse=True)
def _awake_unless_testing_mode(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
):
    """Live ``agent_mode.json`` may be sleep; the suite must not depend on it."""
    if request.module.__name__.endswith("test_agent_mode"):
        return
    monkeypatch.setattr("src.agent_mode.is_asleep", lambda: False)
    monkeypatch.setattr("src.api.telegram_bot.is_asleep", lambda: False, raising=False)


@pytest.fixture
def disposable_postgres(monkeypatch: pytest.MonkeyPatch):
    """Point AgentMemory at TEST_DATABASE_URL only; never silently use DATABASE_URL."""
    if "TEST_DATABASE_URL" not in os.environ:
        pytest.skip(
            "Set TEST_DATABASE_URL (disposable Postgres, same schema as agent) for tests that TRUNCATE."
        )
    assert (
        "TEST_DATABASE_URL" in os.environ
    ), "TRUNCATE tests must read TEST_DATABASE_URL from the environment explicitly."

    url = (os.environ.get("TEST_DATABASE_URL") or "").strip()
    if not url:
        pytest.skip("TEST_DATABASE_URL is present but empty.")

    prod_url = (os.getenv("DATABASE_URL") or settings.database_url or "").strip()
    if prod_url and url == prod_url:
        raise RuntimeError(
            "TEST_DATABASE_URL must not equal DATABASE_URL / production database URL — refusing to TRUNCATE."
        )

    monkeypatch.setattr(settings, "database_url", url)
    monkeypatch.setattr(settings, "test_database_url", url)
    close_pool()
    yield url
    close_pool()


@pytest.fixture
def reset_solitary_state():
    """Opt-in alias: full wipe already clears rhythm + cancel via app_runtime."""
    reset_process_state_for_tests()
    yield
    reset_process_state_for_tests()


@pytest.fixture
def reset_outreach_state(monkeypatch):
    """Opt-in alias: full wipe already clears outreach flags via app_runtime."""
    reset_process_state_for_tests()
    # Live agent_mode.json may be sleep — outreach unit tests assume awake.
    monkeypatch.setattr("src.agent_mode.is_asleep", lambda: False)
    yield
    reset_process_state_for_tests()


@pytest.fixture
def reset_app_runtime():
    """Explicit full wipe (same as autouse; for tests that want a named dependency)."""
    reset_process_state_for_tests()
    yield
    reset_process_state_for_tests()
