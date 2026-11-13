"""Tests for web fetch/search tools."""

from __future__ import annotations

from unittest.mock import MagicMock

import httpx
import pytest

from src.config import settings
from src.tools import web_core as wc
from src.tools import web_tools as wt
from tests.support.web import patch_http_stream


@pytest.fixture(autouse=True)
def _reset_web_core_state():
    wc.clear_dns_public_cache()
    wc.reset_http_client()
    wc.clear_dns_public_cache()
    yield
    wc.clear_dns_public_cache()
    wc.reset_http_client()
    wc.clear_dns_public_cache()


@pytest.mark.unit
def test_html_to_text_and_title():
    html = "<html><head><title>Hello &amp; World</title></head><body><p>Line</p></body></html>"
    assert wc.extract_title(html) == "Hello & World"
    assert "Line" in wc.html_to_text(html)


@pytest.mark.unit
def test_normalize_public_http_url_blocks_localhost():
    with pytest.raises(ValueError, match="not allowed"):
        wc.normalize_public_http_url("http://localhost/secret")


@pytest.mark.unit
def test_normalize_public_http_url_blocks_private_ip():
    with pytest.raises(ValueError, match="not allowed"):
        wc.normalize_public_http_url("http://127.0.0.1/")


@pytest.mark.unit
def test_normalize_public_http_url_ok(monkeypatch):
    monkeypatch.setattr(wc, "_resolve_host_is_public", lambda _h: True)
    out = wc.normalize_public_http_url("https://example.com/path")
    assert out.startswith("https://example.com/")


@pytest.mark.unit
def test_fetch_url_disabled(monkeypatch):
    monkeypatch.setattr(settings, "web_fetch_enabled", False, raising=False)
    assert "disabled" in wc.fetch_url_text("https://example.com")


@pytest.mark.unit
def test_http_client_helpers():
    assert wc._user_agent()
    assert wc._http_timeout().read is not None
    client = wc._http_client()
    assert client is wc._http_client()
    client.close()
    assert wc._http_client() is not client


@pytest.mark.unit
def test_safe_http_get_redirect_missing_location(monkeypatch):
    monkeypatch.setattr(wc, "normalize_public_http_url", lambda u: u)

    class _Redirect:
        status_code = 302
        headers = {}

        def raise_for_status(self):
            raise httpx.HTTPStatusError("302", request=MagicMock(), response=self)

    patch_http_stream(monkeypatch, _Redirect())
    out = wc.fetch_url_text("https://example.com/bad-redirect")
    assert "HTTP 302" in out


@pytest.mark.unit
def test_fetch_url_rejected_normalize_message(monkeypatch):
    monkeypatch.setattr(settings, "web_fetch_enabled", True, raising=False)
    monkeypatch.setattr(
        wc,
        "normalize_public_http_url",
        lambda u: (_ for _ in ()).throw(ValueError("URL host is not allowed")),
    )
    out = wc.fetch_url_text("http://evil")
    assert "URL rejected" in out
    assert "not allowed" in out


@pytest.mark.unit
def test_resolve_host_empty_infos(monkeypatch):
    wc.clear_dns_public_cache()
    monkeypatch.setattr(wc.socket, "getaddrinfo", lambda *a, **k: [])
    assert wc._resolve_host_is_public("fresh-empty-host.example") is False


@pytest.mark.unit
def test_dns_public_cache_expires(monkeypatch):
    import socket

    wc.clear_dns_public_cache()
    calls = {"n": 0}

    def _addrinfo(*_a, **_k):
        calls["n"] += 1
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))]

    monkeypatch.setattr(wc.socket, "getaddrinfo", _addrinfo)
    monkeypatch.setattr(wc, "_DNS_PUBLIC_CACHE_TTL_SEC", 0.0)
    assert wc._resolve_host_is_public("example-cache.test") is True
    assert wc._resolve_host_is_public("example-cache.test") is True
    assert calls["n"] == 2


@pytest.mark.unit
def test_search_lang_detects_cyrillic():
    assert wc._search_lang("dollar rate") == "en"
    assert wc._search_lang("python docs") == "en"


@pytest.mark.unit
def test_resolve_host_empty_string():
    assert wc._resolve_host_is_public("   ") is False


@pytest.mark.unit
def test_reset_http_client_when_already_closed():
    wc._thread_local.client = MagicMock(is_closed=True)
    wc.reset_http_client()
    assert wc._thread_local.client is None


@pytest.mark.unit
def test_fetch_url_happy(monkeypatch):
    monkeypatch.setattr(settings, "web_fetch_enabled", True, raising=False)
    monkeypatch.setattr(wc, "normalize_public_http_url", lambda u: u)

    class _Resp:
        status_code = 200
        headers = {"content-type": "text/html; charset=utf-8"}

        def raise_for_status(self):
            return None

        def iter_bytes(self):
            yield b"<html><title>T</title><body>Body text</body></html>"

    patch_http_stream(monkeypatch, _Resp())
    out = wc.fetch_url_text("https://example.com")
    assert "Body text" in out
    assert "Title: T" in out


@pytest.mark.unit
def test_ip_is_public_rejects_cgnat():
    import ipaddress

    assert wc._ip_is_public(ipaddress.ip_address("100.64.0.1")) is False
    assert wc._ip_is_public(ipaddress.ip_address("8.8.8.8")) is True


@pytest.mark.unit
def test_fetch_url_rejects_unsupported_or_missing_content_type(monkeypatch):
    monkeypatch.setattr(settings, "web_fetch_enabled", True, raising=False)
    monkeypatch.setattr(wc, "normalize_public_http_url", lambda u: u)

    class _Pdf:
        status_code = 200
        headers = {"content-type": "application/pdf"}

        def raise_for_status(self):
            return None

        def iter_bytes(self):
            yield b"%PDF-1.4 binary junk"

    patch_http_stream(monkeypatch, _Pdf())
    out = wc.fetch_url_text("https://example.com/doc.pdf")
    assert "unsupported content-type" in out

    class _NoCt:
        status_code = 200
        headers = {}

        def raise_for_status(self):
            return None

        def iter_bytes(self):
            yield b"<html><body>secret</body></html>"

    patch_http_stream(monkeypatch, _NoCt())
    out2 = wc.fetch_url_text("https://example.com/noct")
    assert "unsupported content-type" in out2
    assert "secret" not in out2


@pytest.mark.unit
def test_fetch_url_http_error(monkeypatch):
    monkeypatch.setattr(settings, "web_fetch_enabled", True, raising=False)
    monkeypatch.setattr(wc, "normalize_public_http_url", lambda u: u)

    class _Resp:
        status_code = 404

        def raise_for_status(self):
            raise httpx.HTTPStatusError("404", request=MagicMock(), response=self)

    patch_http_stream(monkeypatch, _Resp())
    out = wc.fetch_url_text("https://example.com/missing")
    assert "HTTP 404" in out


@pytest.mark.unit
def test_search_web_disabled(monkeypatch):
    monkeypatch.setattr(settings, "web_search_enabled", False, raising=False)
    assert "disabled" in wc.search_web_results("rust lang")


@pytest.mark.unit
def test_search_web_happy(monkeypatch):
    monkeypatch.setattr(settings, "web_search_enabled", True, raising=False)
    monkeypatch.setattr(
        wc,
        "_web_text_search",
        lambda q, max_results=5: [
            {
                "title": "Rust",
                "href": "https://rust-lang.org",
                "body": "Systems language",
            }
        ],
    )
    out = wc.search_web_results("rust programming")
    assert "Rust" in out
    assert "rust-lang.org" in out


@pytest.mark.unit
def test_tool_wrappers(monkeypatch):
    monkeypatch.setattr(wt, "search_web_results", lambda q: f"hits:{q}")
    monkeypatch.setattr(wt, "fetch_url_text", lambda u: f"page:{u}")
    assert wt.search_web.invoke({"query": "ab"}) == "hits:ab"
    assert wt.fetch_url.invoke({"url": "https://a.b"}) == "page:https://a.b"


@pytest.mark.unit
def test_resolve_host_oserror_and_empty(monkeypatch):
    monkeypatch.setattr(
        wc.socket, "getaddrinfo", lambda *a, **k: (_ for _ in ()).throw(OSError())
    )
    assert wc._resolve_host_is_public("example.com") is False
    monkeypatch.setattr(wc.socket, "getaddrinfo", lambda *a, **k: [])
    assert wc._resolve_host_is_public("example.com") is False


@pytest.mark.unit
def test_resolve_host_is_public_blocks(monkeypatch):
    monkeypatch.setattr(
        wc.socket, "getaddrinfo", lambda *a, **k: [(2, 1, 6, "", ("127.0.0.1", 0))]
    )
    assert wc._resolve_host_is_public("evil.example") is False


@pytest.mark.unit
def test_resolve_host_is_public_ok(monkeypatch):
    monkeypatch.setattr(
        wc.socket,
        "getaddrinfo",
        lambda *a, **k: [(2, 1, 6, "", ("93.184.216.34", 0))],
    )
    assert wc._resolve_host_is_public("example.com") is True


@pytest.mark.unit
def test_resolve_host_only_invalid_addresses(monkeypatch):
    monkeypatch.setattr(
        wc.socket,
        "getaddrinfo",
        lambda *a, **k: [(2, 1, 6, "", ("not-ip", 0))],
    )
    assert wc._resolve_host_is_public("example.com") is False


@pytest.mark.unit
def test_fetch_url_too_large(monkeypatch):
    monkeypatch.setattr(settings, "web_fetch_enabled", True, raising=False)
    monkeypatch.setattr(settings, "web_fetch_max_bytes", 10, raising=False)
    monkeypatch.setattr(wc, "normalize_public_http_url", lambda u: u)

    class _Resp:
        status_code = 200
        headers = {"content-type": "text/plain"}

        def raise_for_status(self):
            return None

        def iter_bytes(self):
            yield b"x" * 20

    patch_http_stream(monkeypatch, _Resp())
    out = wc.fetch_url_text("https://example.com/big")
    assert "too large" in out


@pytest.mark.unit
def test_reset_http_client_closes_thread_local(monkeypatch):
    closed = {"ok": False}

    class _Client:
        is_closed = False

        def close(self):
            closed["ok"] = True
            self.is_closed = True

    client = _Client()
    wc._thread_local.client = client
    wc.reset_http_client()
    assert closed["ok"] is True
    assert wc._thread_local.client is None


@pytest.mark.unit
def test_pinned_fetch_targets(monkeypatch):
    monkeypatch.setattr(
        wc, "_resolve_public_ips_uncached", lambda _h: ["93.184.216.34"]
    )
    pinned, headers = wc._pinned_fetch_targets("https://example.com/path")
    assert pinned.startswith("https://93.184.216.34")
    assert headers["Host"] == "example.com"


@pytest.mark.unit
def test_pinned_fetch_targets_ipv6_and_custom_port(monkeypatch):
    monkeypatch.setattr(wc, "_resolve_public_ips_uncached", lambda _h: ["2001:db8::1"])
    pinned, headers = wc._pinned_fetch_targets("https://example.com:8443/x")
    assert pinned.startswith("https://[2001:db8::1]:8443")
    assert headers["Host"] == "example.com"


@pytest.mark.unit
def test_pinned_fetch_targets_no_public_ips(monkeypatch):
    monkeypatch.setattr(wc, "_resolve_public_ips_uncached", lambda _h: [])
    with pytest.raises(ValueError, match="private or local"):
        wc._pinned_fetch_targets("https://example.com/")


@pytest.mark.unit
def test_pinned_fetch_targets_missing_host():
    with pytest.raises(ValueError, match="not allowed"):
        wc._pinned_fetch_targets("https:///path")


@pytest.mark.unit
def test_safe_http_get_pins_host_header(monkeypatch):
    monkeypatch.setattr(wc, "normalize_public_http_url", lambda u: u)
    monkeypatch.setattr(
        wc, "_resolve_public_ips_uncached", lambda _h: ["93.184.216.34"]
    )

    seen: dict[str, object] = {}

    class _Resp:
        status_code = 200
        headers = {"content-type": "text/plain"}

        def raise_for_status(self):
            return None

        def iter_bytes(self):
            yield b"ok"

    class _Client:
        is_closed = False

        def stream(self, method, url, headers=None, extensions=None):
            seen["url"] = url
            seen["headers"] = headers
            seen["extensions"] = extensions
            cm = MagicMock()
            cm.__enter__ = MagicMock(return_value=_Resp())
            cm.__exit__ = MagicMock(return_value=False)
            return cm

    monkeypatch.setattr(wc, "_http_client", lambda: _Client())
    wc._safe_http_get("https://example.com/")
    assert seen["url"] == "https://93.184.216.34/"
    assert seen["headers"]["Host"] == "example.com"
    assert seen["extensions"]["sni_hostname"] == "example.com"


@pytest.mark.unit
def test_safe_http_get_follows_redirect(monkeypatch):
    monkeypatch.setattr(wc, "normalize_public_http_url", lambda u: u)
    monkeypatch.setattr(
        wc, "_resolve_public_ips_uncached", lambda _h: ["93.184.216.34"]
    )

    class _Redirect:
        status_code = 302
        headers = {"location": "/final"}

        def raise_for_status(self):
            return None

        def iter_bytes(self):
            return iter([])

    class _Final:
        status_code = 200
        headers = {"content-type": "text/plain"}

        def raise_for_status(self):
            return None

        def iter_bytes(self):
            yield b"done"

    responses = [_Redirect(), _Final()]

    class _Client:
        is_closed = False

        def stream(self, method, url, headers=None, extensions=None):
            resp = responses.pop(0)
            cm = MagicMock()
            cm.__enter__ = MagicMock(return_value=resp)
            cm.__exit__ = MagicMock(return_value=False)
            return cm

    monkeypatch.setattr(wc, "_http_client", lambda: _Client())
    safe_url, headers, raw = wc._safe_http_get("https://example.com/start")
    assert safe_url.endswith("/final")
    assert raw == b"done"


@pytest.mark.unit
def test_safe_http_get_blocks_redirect_to_private(monkeypatch):
    calls: list[str] = []

    def _normalize(url: str) -> str:
        calls.append(url)
        if "127.0.0.1" in url:
            raise ValueError("URL host is not allowed")
        return url

    monkeypatch.setattr(wc, "normalize_public_http_url", _normalize)
    monkeypatch.setattr(
        wc, "_resolve_public_ips_uncached", lambda _h: ["93.184.216.34"]
    )

    class _Redirect:
        status_code = 302
        headers = {"location": "http://127.0.0.1/secret"}

        def raise_for_status(self):
            return None

        def iter_bytes(self):
            return iter([])

    patch_http_stream(monkeypatch, _Redirect())
    with pytest.raises(ValueError, match="not allowed"):
        wc._safe_http_get("https://example.com/redirect")


@pytest.mark.unit
def test_safe_http_get_too_many_redirects(monkeypatch):
    monkeypatch.setattr(wc, "normalize_public_http_url", lambda u: u)
    monkeypatch.setattr(
        wc, "_resolve_public_ips_uncached", lambda _h: ["93.184.216.34"]
    )

    class _Redirect:
        status_code = 302
        headers = {"location": "/again"}

        def raise_for_status(self):
            return None

        def iter_bytes(self):
            return iter([])

    patch_http_stream(monkeypatch, _Redirect())
    with pytest.raises(ValueError, match="too many redirects"):
        wc._safe_http_get("https://example.com/loop")


@pytest.mark.unit
def test_search_web_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr(settings, "web_search_enabled", True, raising=False)
    monkeypatch.setattr(settings, "web_search_max_retries", 3, raising=False)
    monkeypatch.setattr(wc.time, "sleep", lambda _s: None)

    attempts = {"n": 0}

    def _search(q, max_results=5):
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise RuntimeError("rate limited")
        return [{"title": "Ok", "href": "https://ok", "body": "found"}]

    monkeypatch.setattr(wc, "_web_text_search", _search)
    out = wc.search_web_results("retry query here")
    assert attempts["n"] == 2
    assert "Ok" in out


@pytest.mark.unit
def test_web_core_edge_cases(monkeypatch):
    assert wc._blocked_host("") is True
    assert wc._blocked_host("metadata.google.internal") is True
    assert wc._blocked_host("host.local") is True
    assert wc._blocked_host("10.0.0.1") is True
    assert wc._blocked_host("169.254.0.1") is True
    assert wc._blocked_host("example.com") is False
    assert wc._blocked_host("svc.internal") is True
    assert wc._resolve_host_is_public("localhost") is False
    monkeypatch.setattr(
        wc.socket,
        "getaddrinfo",
        lambda *a, **k: [
            (2, 1, 6, "", ("not-ip", 0)),
            (2, 1, 6, "", ("93.184.216.34", 0)),
        ],
    )
    assert wc._resolve_host_is_public("example.com") is True
    monkeypatch.setattr(wc, "_resolve_host_is_public", lambda _h: True)
    assert wc.normalize_public_http_url("//example.com/path").startswith("https://")
    with pytest.raises(ValueError):
        wc.normalize_public_http_url("")
    with pytest.raises(ValueError):
        wc.normalize_public_http_url("ftp://example.com")
    with pytest.raises(ValueError):
        wc.normalize_public_http_url("https://example.com:22/")
    monkeypatch.setattr(wc, "_resolve_host_is_public", lambda _h: False)
    with pytest.raises(ValueError, match="private"):
        wc.normalize_public_http_url("https://evil.example/")

    monkeypatch.setattr(
        wc.socket, "getaddrinfo", lambda *a, **k: (_ for _ in ()).throw(OSError())
    )
    assert wc._resolve_host_is_public("bad.host") is False
    monkeypatch.setattr(wc.socket, "getaddrinfo", lambda *a, **k: [])
    assert wc._resolve_host_is_public("empty.host") is False

    assert wc.html_to_text("", max_chars=0) == ""
    long = "word " * 4000
    assert long.endswith(" ")
    trimmed = wc.html_to_text(long, max_chars=100)
    assert trimmed.endswith("…")
    assert len(trimmed) == 101
    assert wc.extract_title("<html></html>") == ""

    monkeypatch.setattr(settings, "web_search_enabled", True, raising=False)
    assert "too short" in wc.search_web_results("x")

    long_calls: list[str] = []

    def _long_search(q, max_results=5):
        long_calls.append(q)
        return [{"title": "T", "href": "https://x", "body": "b"}]

    monkeypatch.setattr(wc, "_web_text_search", _long_search)
    assert wc.search_web_results("a" * 600).startswith("Web search:")
    assert len(long_calls[0]) <= 160

    monkeypatch.setattr(wc, "_web_text_search", lambda q, max_results=5: [])
    assert "unavailable" in wc.search_web_results("empty query here").lower()

    attempts = {"n": 0}

    def _rate_limit(q, max_results=5):
        attempts["n"] += 1
        raise ValueError("search rate-limited (202)")

    monkeypatch.setattr(wc, "_web_text_search", _rate_limit)
    out = wc.search_web_results("something searchable", max_retries=3)
    assert attempts["n"] == 3
    assert "unavailable" in out.lower()

    monkeypatch.setattr(
        wc,
        "_web_text_search",
        lambda q, max_results=5: (_ for _ in ()).throw(RuntimeError("net down")),
    )
    assert "unavailable" in wc.search_web_results("something searchable").lower()

    monkeypatch.setattr(settings, "web_fetch_enabled", True, raising=False)
    monkeypatch.setattr(wc, "normalize_public_http_url", lambda u: u)

    class _EmptyResp:
        status_code = 200
        headers = {"content-type": "text/html"}

        def raise_for_status(self):
            return None

        def iter_bytes(self):
            return iter([])

    patch_http_stream(monkeypatch, _EmptyResp())
    assert "no readable text" in wc.fetch_url_text("https://example.com/empty")

    class _BadCharsetResp(_EmptyResp):
        headers = {"content-type": "text/html; charset=not-real-charset"}

        def iter_bytes(self):
            yield b""
            yield b"<html><body>ok</body></html>"

    patch_http_stream(monkeypatch, _BadCharsetResp())
    out = wc.fetch_url_text("https://example.com/charset")
    assert "ok" in out

    class _BoomClient:
        is_closed = False

        def stream(self, method, url, headers=None, extensions=None):
            raise RuntimeError("boom")

    monkeypatch.setattr(
        wc, "_resolve_public_ips_uncached", lambda _h: ["93.184.216.34"]
    )
    monkeypatch.setattr(wc, "_http_client", lambda: _BoomClient())
    assert "Fetch failed" in wc.fetch_url_text("https://example.com/err")

    monkeypatch.setattr(
        wc,
        "normalize_public_http_url",
        lambda u: (_ for _ in ()).throw(ValueError("bad")),
    )
    assert "URL rejected" in wc.fetch_url_text("http://bad")


@pytest.mark.unit
def test_search_http_client_helpers():
    client = wc._search_http_client()
    assert client is wc._search_http_client()
    client.close()
    assert wc._search_http_client() is not client
    wc.reset_search_http_client()


@pytest.mark.unit
def test_searxng_loopback_guard():
    with pytest.raises(ValueError, match="loopback"):
        wc._assert_loopback_searxng("https://searx.example.org")
    assert wc._assert_loopback_searxng("http://127.0.0.1:8888").endswith("8888")


@pytest.mark.unit
def test_web_text_search_chain_order(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "src.tools.search_cache.settings.web_search_cache_path",
        str(tmp_path / "c.sqlite"),
        raising=False,
    )
    from src.tools import search_cache as sc

    sc.reset_search_cache_for_tests()
    order: list[str] = []

    def _sx(q, max_results, lang):
        order.append("searxng")
        return []

    def _dd(q, max_results, lang):
        order.append("ddgs")
        return [{"title": "D", "href": "https://d.test", "body": "b"}]

    def _pw(q, max_results):
        order.append("playwright_serp")
        raise AssertionError("should stop at ddgs")

    monkeypatch.setattr(wc, "_searxng_rows", _sx)
    monkeypatch.setattr(wc, "_ddgs_rows", _dd)
    monkeypatch.setattr(wc, "_playwright_ddg_serp_rows", _pw)
    out = wc._web_text_search("chain order", max_results=2)
    assert out[0]["href"] == "https://d.test"
    assert order == ["searxng", "ddgs"]
    assert wc.last_search_backend() == "ddgs"
    sc.reset_search_cache_for_tests()


@pytest.mark.unit
def test_search_title_and_user_agent_helpers(monkeypatch):
    assert wc._clean_search_title(".css-x{color:red}Real Title") == "Real Title"
    monkeypatch.setattr(
        settings, "web_search_user_agent", "CustomBot/2.0", raising=False
    )
    assert wc._search_user_agent() == "CustomBot/2.0"


@pytest.mark.unit
def test_check_searxng_health_ok(monkeypatch):
    class _Resp:
        status_code = 200

        def json(self):
            return {"results": []}

    class _Client:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, *a, **k):
            return _Resp()

    monkeypatch.setattr(wc.httpx, "Client", _Client)
    monkeypatch.setattr(
        wc.settings, "searxng_url", "http://127.0.0.1:8888", raising=False
    )
    assert wc.check_searxng_health() is True


@pytest.mark.unit
def test_check_searxng_health_warns(monkeypatch):
    class _Client:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, *a, **k):
            raise wc.httpx.ConnectError("down")

    monkeypatch.setattr(wc.httpx, "Client", _Client)
    assert wc.check_searxng_health() is False
