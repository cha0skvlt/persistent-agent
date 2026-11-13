"""Browser request guard: every subresource is gated, not just the top page."""

from __future__ import annotations

from urllib.parse import urlparse

import pytest

from src.tools import browse_web as bw

_PRIVATE_HOSTS = {"127.0.0.1", "localhost", "169.254.169.254", "10.0.0.5", "::1"}


class _Req:
    def __init__(self, url: str) -> None:
        self.url = url


class _Route:
    def __init__(self, url: str) -> None:
        self.request = _Req(url)
        self.aborted = False
        self.continued = False

    def abort(self) -> None:
        self.aborted = True

    def continue_(self) -> None:
        self.continued = True


class _Page:
    def __init__(self) -> None:
        self.handler = None

    def route(self, pattern: str, handler) -> None:
        assert pattern == "**/*"
        self.handler = handler


@pytest.fixture
def guard(monkeypatch):
    def _fake_assert(url: str) -> str:
        host = (urlparse(url).hostname or "").lower()
        if host in _PRIVATE_HOSTS:
            raise ValueError(f"blocked host {host}")
        return url

    monkeypatch.setattr(bw, "_assert_public_url", _fake_assert)
    page = _Page()
    bw._install_request_guard(page)
    assert page.handler is not None
    return page.handler


@pytest.mark.unit
@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/app.js",
        "https://cdn.example.com/img.png",
        "https://example.com/api/data.json",
        "data:image/png;base64,AAAA",
        "blob:https://example.com/abc",
        "about:blank",
    ],
)
def test_public_and_local_schemes_continue(guard, url):
    r = _Route(url)
    guard(r)
    assert r.continued is True
    assert r.aborted is False


@pytest.mark.unit
@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8080/admin",
        "http://localhost:5432/",
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.5/internal",
        "http://[::1]:8000/",
    ],
)
def test_private_targets_abort(guard, url):
    r = _Route(url)
    guard(r)
    assert r.aborted is True
    assert r.continued is False


@pytest.mark.unit
def test_guard_aborts_when_request_url_is_unreadable(guard):
    class _Broken:
        @property
        def url(self):
            raise RuntimeError("gone")

    class _R:
        request = _Broken()
        aborted = False
        continued = False

        def abort(self):
            self.aborted = True

        def continue_(self):
            self.continued = True

    r = _R()
    guard(r)
    assert r.aborted is True
    assert r.continued is False
