"""Local browse_web tool — mocked session, fail-closed paths."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest

from src.config import settings
from src.tools import browse_web as bw

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clean_session():
    bw.close_browse_session()
    yield
    bw.close_browse_session()


def test_browse_disabled(monkeypatch):
    monkeypatch.setattr(settings, "web_browse_enabled", False, raising=False)
    assert "disabled" in bw.browse_web_impl(action="open", target="https://example.com")


def test_browse_requires_open_before_follow():
    out = bw.browse_web_impl(action="follow", target="/next")
    assert "no session" in out


def test_browse_bad_action():
    assert "action must be" in bw.browse_web_impl(action="dance", target="")
    assert "action must be" in bw.browse_web_impl(action="links", target="")


def _fake_page(contents: dict[str, str], start_url: str) -> MagicMock:
    page = MagicMock()
    page.url = start_url

    def _content():
        return contents.get(page.url, next(iter(contents.values())))

    def _title():
        return "Board" if "/res/" not in page.url else "Thread"

    def _goto(url, wait_until="domcontentloaded"):
        page.url = url
        return None

    def _inner_text(sel="body"):
        from src.tools.web_extract import extract_page_text

        return extract_page_text(_content(), max_chars=0)

    page.content.side_effect = _content
    page.title.side_effect = _title
    page.goto.side_effect = _goto
    page.inner_text.side_effect = _inner_text
    page.wait_for_load_state = MagicMock()
    page.set_default_timeout = MagicMock()
    return page


def test_browse_open_follow_extract(monkeypatch):
    monkeypatch.setattr(settings, "web_browse_enabled", True, raising=False)
    monkeypatch.setattr(settings, "web_browse_max_steps", 8, raising=False)
    monkeypatch.setattr(
        "src.tools.web_core._resolve_host_is_public",
        lambda _h: True,
    )

    html_board = (
        "<html><head><title>Board</title></head><body>"
        "<p>Board index</p>"
        "<div>Section one topic</div>"
        "<div>Section two topic</div>"
        "</body></html>"
    )
    html_thread = (
        "<html><head><title>Thread</title></head><body>"
        "<p>Post body here</p>"
        "</body></html>"
    )
    contents = {
        "https://example.com/b/": html_board,
        "https://example.com/res/9.html": html_thread,
    }
    page = _fake_page(contents, "https://example.com/b/")

    def _ensure():
        sess = bw._BrowseSession()
        sess.browser = MagicMock()
        sess.playwright = MagicMock()
        sess.page = page
        bw._session = sess
        return sess

    monkeypatch.setattr(bw, "_ensure_session", _ensure)

    out = bw.browse_web_impl(action="open", target="https://example.com/b/")
    assert "Board index" in out
    assert "Section one topic" in out
    assert "Section two topic" in out
    assert "Links:" not in out

    out2 = bw.browse_web_impl(action="follow", target="https://example.com/res/9.html")
    assert "Post body" in out2

    closed = bw.browse_web_impl(action="close")
    assert closed == "browse_closed"


def test_browse_follow_rejects_index(monkeypatch):
    monkeypatch.setattr(settings, "web_browse_enabled", True, raising=False)
    monkeypatch.setattr(
        "src.tools.web_core._resolve_host_is_public",
        lambda _h: True,
    )
    page = _fake_page(
        {"https://example.com/": "<html><body><p>hi</p></body></html>"},
        "https://example.com/",
    )

    def _ensure():
        sess = bw._BrowseSession()
        sess.browser = MagicMock()
        sess.playwright = MagicMock()
        sess.page = page
        bw._session = sess
        return sess

    monkeypatch.setattr(bw, "_ensure_session", _ensure)
    bw.browse_web_impl(action="open", target="https://example.com/")
    out = bw.browse_web_impl(action="follow", target="1")
    assert "not a link index" in out


def test_browse_crash_fail_closed(monkeypatch):
    monkeypatch.setattr(settings, "web_browse_enabled", True, raising=False)

    def _boom():
        raise RuntimeError("chromium missing")

    monkeypatch.setattr(bw, "_ensure_session", _boom)
    out = bw.browse_web_impl(action="open", target="https://example.com/")
    assert out.startswith("browse_error:")
    assert bw._session is None


def test_browse_step_cap(monkeypatch):
    monkeypatch.setattr(settings, "web_browse_enabled", True, raising=False)
    monkeypatch.setattr(settings, "web_browse_max_steps", 1, raising=False)
    page = MagicMock()
    page.url = "https://example.com/"
    page.content.return_value = "<html><body><article>x</article></body></html>"
    page.inner_text.return_value = "x"
    page.title.return_value = "t"
    sess = bw._BrowseSession()
    sess.browser = MagicMock()
    sess.playwright = MagicMock()
    sess.page = page
    sess.steps = 1
    bw._session = sess
    out = bw.browse_web_impl(action="extract")
    assert "step cap" in out


def test_browse_close_from_other_thread(monkeypatch):
    """close_browse_session must not raise when called off the Playwright worker."""
    monkeypatch.setattr(settings, "web_browse_enabled", True, raising=False)
    monkeypatch.setattr(
        "src.tools.web_core._resolve_host_is_public",
        lambda _h: True,
    )
    page = MagicMock()
    page.url = "https://example.com/"
    page.content.return_value = "<html><body><article>hi</article></body></html>"
    page.inner_text.return_value = "hi"
    page.title.return_value = "t"
    page.goto.side_effect = lambda url, wait_until="domcontentloaded": setattr(
        page, "url", url
    )
    page.set_default_timeout = MagicMock()

    def _ensure():
        sess = bw._BrowseSession()
        sess.browser = MagicMock()
        sess.playwright = MagicMock()
        sess.page = page
        bw._session = sess
        return sess

    monkeypatch.setattr(bw, "_ensure_session", _ensure)
    assert "hi" in bw.browse_web_impl(action="open", target="https://example.com/")
    err: list[BaseException] = []

    def _close():
        try:
            bw.close_browse_session()
        except BaseException as exc:  # noqa: BLE001
            err.append(exc)

    t = threading.Thread(target=_close)
    t.start()
    t.join(timeout=5)
    assert not err
    assert bw._session is None
