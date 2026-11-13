"""Shared httpx / DNS mocks for ``web_core`` tests."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.tools import web_core as wc

DEFAULT_PIN_IP = "93.184.216.34"


def patch_public_dns(monkeypatch, ip: str = DEFAULT_PIN_IP) -> None:
    monkeypatch.setattr(
        wc,
        "_resolve_public_ips_uncached",
        lambda host: [ip] if (host or "").strip() else [],
    )


def patch_http_stream(monkeypatch, resp, *, pin_ip: str = DEFAULT_PIN_IP) -> None:
    """Install a thread-local client whose ``stream()`` yields ``resp``."""
    patch_public_dns(monkeypatch, pin_ip)

    class _Client:
        is_closed = False

        def stream(self, method, url, headers=None, extensions=None):
            cm = MagicMock()
            cm.__enter__ = MagicMock(return_value=resp)
            cm.__exit__ = MagicMock(return_value=False)
            return cm

    monkeypatch.setattr(wc, "_http_client", lambda: _Client())
