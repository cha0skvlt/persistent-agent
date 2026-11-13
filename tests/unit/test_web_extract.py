"""Full page extract for fetch_url / browse_web — any site."""

from __future__ import annotations

import pytest

from src.tools import web_core as wc
from src.tools import web_extract as we

pytestmark = pytest.mark.unit


def test_extract_full_page_keeps_whole_body():
    html = """
    <html><body>
      <nav>Home About</nav>
      <div><p>First block of content.</p></div>
      <div><p>Second block of content.</p></div>
      <footer>copyright</footer>
    </body></html>
    """
    text = we.extract_page_text(html, max_chars=14000)
    assert "First block of content" in text
    assert "Second block of content" in text
    assert "Home About" in text


def test_compact_visible_text_kills_blank_line_bombs():
    bomb = "Title\n" + ("\n" * 80) + "Real paragraph\n" + ("\n" * 40) + "End"
    out = we.compact_visible_text(bomb)
    assert out == "Title\n\nReal paragraph\n\nEnd"
    assert out.count("\n\n\n") == 0


def test_fetch_url_returns_full_page(monkeypatch):
    monkeypatch.setattr(wc, "_resolve_host_is_public", lambda _h: True)
    html = (
        "<html><head><title>Page</title></head>"
        "<body><p>Hello page</p>"
        "<div>Section alpha</div>"
        "<div>Section beta</div>"
        "</body></html>"
    )

    def _fake_get(url: str):
        return url, {"content-type": "text/html; charset=utf-8"}, html.encode()

    monkeypatch.setattr(wc, "_safe_http_get", _fake_get)
    monkeypatch.setattr(wc.settings, "web_fetch_enabled", True, raising=False)
    out = wc.fetch_url_text("https://example.com/page/")
    assert "Hello page" in out
    assert "Section alpha" in out
    assert "Section beta" in out
    assert "Links:" not in out


def test_redirect_note_path_change():
    assert we.redirect_note("https://a.example/b/", "https://a.example/")
    assert not we.redirect_note("https://a.example/b/", "https://a.example/b/")


def test_fetch_url_redirect_note(monkeypatch):
    monkeypatch.setattr(wc, "_resolve_host_is_public", lambda _h: True)
    html = "<html><body><p>Home</p></body></html>"

    def _fake_get(url: str):
        return (
            "https://example.com/",
            {"content-type": "text/html; charset=utf-8"},
            html.encode(),
        )

    monkeypatch.setattr(wc, "_safe_http_get", _fake_get)
    monkeypatch.setattr(wc.settings, "web_fetch_enabled", True, raising=False)
    out = wc.fetch_url_text("https://example.com/b/")
    assert "redirected from https://example.com/b/" in out
    assert "URL: https://example.com/" in out


def test_fetch_url_json_returns_raw_body(monkeypatch):
    monkeypatch.setattr(wc, "_resolve_host_is_public", lambda _h: True)
    payload = b'{"ok": true, "n": 42}'

    def _fake_get(url: str):
        return url, {"content-type": "application/json"}, payload

    monkeypatch.setattr(wc, "_safe_http_get", _fake_get)
    monkeypatch.setattr(wc.settings, "web_fetch_enabled", True, raising=False)
    out = wc.fetch_url_text("https://example.com/api.json")
    assert '"ok": true' in out
    assert "Title:" not in out
