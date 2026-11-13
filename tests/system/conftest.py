"""System tier — API, heart, imports, optional live Ollama."""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _skip_live_boot_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    """CI and default test runs do not require Ollama/Postgres at API startup."""
    if os.getenv("SKIP_LIVE_TESTS", "").strip() not in ("1", "true", "yes"):
        return
    try:
        import src.api.app as api_app

        monkeypatch.setattr(
            api_app, "verify_boot_dependencies_with_retry", lambda: None
        )
    except ImportError:
        pass


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    if os.getenv("SKIP_LIVE_TESTS", "").strip() in ("1", "true", "yes"):
        skip = pytest.mark.skip(reason="SKIP_LIVE_TESTS=1")
        for item in items:
            if "live" in item.keywords:
                item.add_marker(skip)
