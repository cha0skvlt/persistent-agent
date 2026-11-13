"""Hard contract for search_web — chain, output shape, cache, fail-closed.

These tests must fail if search stops working, skips the primary backend,
returns a success-shaped empty body, or caches last-resort stubs.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from src.config import settings
from src.tools import search_cache as sc
from src.tools import web_core as wc
from src.tools import web_tools as wt
from src.tools.web_evidence import web_search_body_usable

pytestmark = pytest.mark.unit

_HIT = {"title": "Example Hit", "href": "https://example.com/hit", "body": "snippet"}


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sc.settings, "web_search_cache_path", str(tmp_path / "search_cache.sqlite")
    )
    monkeypatch.setattr(sc.settings, "web_search_cache_ttl_sec", 86400.0, raising=False)
    monkeypatch.setattr(settings, "web_search_enabled", True, raising=False)
    monkeypatch.setattr(settings, "web_search_max_results", 5, raising=False)
    monkeypatch.setattr(settings, "web_search_max_retries", 1, raising=False)
    monkeypatch.setattr(wc.time, "sleep", lambda _s: None)
    sc.reset_search_cache_for_tests()
    wc._set_last_search_backend("")
    yield
    sc.reset_search_cache_for_tests()
    wc._set_last_search_backend("")


def _stub_chain(
    monkeypatch,
    *,
    searxng: Any = None,
    ddgs: Any = None,
    playwright: Any = None,
    wikipedia: Any = None,
    order: list[str] | None = None,
) -> list[str]:
    """Patch backends. Callables may raise or return rows. Track call order."""
    seen = order if order is not None else []

    def _wrap(name: str, value: Any, takes_lang: bool):
        def _fn(*args, **kwargs):
            seen.append(name)
            if callable(value) and not isinstance(value, list):
                return value(*args, **kwargs)
            if isinstance(value, BaseException):
                raise value
            return list(value or [])

        return _fn

    monkeypatch.setattr(
        wc,
        "_searxng_rows",
        _wrap("searxng", [] if searxng is None else searxng, True),
    )
    monkeypatch.setattr(
        wc, "_ddgs_rows", _wrap("ddgs", [] if ddgs is None else ddgs, True)
    )
    monkeypatch.setattr(
        wc,
        "_playwright_ddg_serp_rows",
        _wrap("playwright_serp", [] if playwright is None else playwright, False),
    )
    monkeypatch.setattr(
        wc,
        "_wikipedia_opensearch_rows",
        _wrap("wikipedia", [] if wikipedia is None else wikipedia, False),
    )
    return seen


# --- chain order / fallbacks -------------------------------------------------


def test_chain_order_is_searxng_ddgs_playwright_wikipedia(monkeypatch):
    order: list[str] = []
    _stub_chain(
        monkeypatch,
        order=order,
        wikipedia=[_HIT],
    )
    rows = wc._web_text_search("contract chain", max_results=3)
    assert rows == [_HIT]
    assert order == ["searxng", "ddgs", "playwright_serp", "wikipedia"]
    assert wc.last_search_backend() == "wikipedia"


def test_searxng_primary_skips_later_backends(monkeypatch):
    order: list[str] = []
    _stub_chain(monkeypatch, order=order, searxng=[_HIT], ddgs=[_HIT])
    rows = wc._web_text_search("primary", max_results=3)
    assert rows[0]["href"] == _HIT["href"]
    assert order == ["searxng"]
    assert wc.last_search_backend() == "searxng"


def test_empty_searxng_falls_to_ddgs(monkeypatch):
    order: list[str] = []
    _stub_chain(monkeypatch, order=order, searxng=[], ddgs=[_HIT])
    assert wc._web_text_search("q", max_results=2)[0]["title"] == "Example Hit"
    assert order == ["searxng", "ddgs"]
    assert wc.last_search_backend() == "ddgs"


def test_exception_searxng_falls_to_ddgs(monkeypatch):
    order: list[str] = []
    _stub_chain(
        monkeypatch,
        order=order,
        searxng=RuntimeError("searxng down"),
        ddgs=[_HIT],
    )
    assert wc._web_text_search("q", max_results=2)
    assert order == ["searxng", "ddgs"]
    assert wc.last_search_backend() == "ddgs"


def test_rate_limited_backend_continues(monkeypatch):
    order: list[str] = []
    _stub_chain(
        monkeypatch,
        order=order,
        searxng=ValueError("search rate-limited (429)"),
        ddgs=ValueError("captcha"),
        playwright=[_HIT],
    )
    rows = wc._web_text_search("q", max_results=2)
    assert rows[0]["href"] == _HIT["href"]
    assert order == ["searxng", "ddgs", "playwright_serp"]
    assert wc.last_search_backend() == "playwright_serp"


def test_all_backends_empty_returns_empty_list(monkeypatch):
    _stub_chain(monkeypatch)
    assert wc._web_text_search("nothing", max_results=3) == []
    assert wc.last_search_backend() == ""


# --- searxng safety / parsing ------------------------------------------------


def test_searxng_rejects_non_loopback_url(monkeypatch):
    monkeypatch.setattr(
        wc.settings, "searxng_url", "https://searx.example.org", raising=False
    )
    with pytest.raises(ValueError, match="loopback"):
        wc._searxng_rows("q", max_results=3, lang="en")


def test_searxng_rows_parse_json_and_skip_empty_href(monkeypatch):
    payload = {
        "results": [
            {"title": "Bad", "url": "", "content": "no"},
            {
                "title": "Good",
                "url": "https://good.example/a",
                "content": "body",
                "engine": "google",
            },
            "not-a-dict",
            {
                "title": "Extra",
                "url": "https://extra.example/b",
                "snippet": "s",
            },
        ]
    }

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return payload

    class _Client:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url, params=None):
            assert "127.0.0.1" in url or "localhost" in url
            assert params["format"] == "json"
            assert params["q"] == "parse me"
            return _Resp()

    monkeypatch.setattr(wc.httpx, "Client", _Client)
    monkeypatch.setattr(
        wc.settings, "searxng_url", "http://127.0.0.1:8888", raising=False
    )
    rows = wc._searxng_rows("parse me", max_results=2, lang="en")
    assert len(rows) == 2
    assert rows[0]["href"] == "https://good.example/a"
    assert rows[0]["engine"] == "google"
    assert rows[1]["href"] == "https://extra.example/b"


def test_searxng_403_json_forbidden_raises(monkeypatch):
    class _Resp:
        status_code = 403

        def raise_for_status(self):
            raise AssertionError("must fail closed before raise_for_status")

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
    with pytest.raises(ValueError, match="json format forbidden"):
        wc._searxng_rows("q", max_results=3, lang="en")


# --- output contract for the LLM --------------------------------------------


def test_search_web_results_format_is_usable_evidence(monkeypatch):
    monkeypatch.setattr(
        wc,
        "_web_text_search",
        lambda q, max_results: [
            {
                "title": "Hit One",
                "href": "https://a.example/1",
                "body": "alpha",
                "engine": "duckduckgo",
            },
            {"title": "Hit Two", "href": "https://b.example/2", "body": "beta"},
        ],
    )
    wc._set_last_search_backend("searxng")
    out = wc.search_web_results("focused query here", max_retries=1)
    assert out.startswith("Web search:")
    assert "backend=searxng" in out
    assert "1. Hit One" in out
    assert "URL: https://a.example/1" in out
    assert "engine: duckduckgo" in out
    assert "2. Hit Two" in out
    assert "URL: https://b.example/2" in out
    assert web_search_body_usable(out) is True


def test_success_output_never_looks_like_failure(monkeypatch):
    monkeypatch.setattr(
        wc, "_web_text_search", lambda q, max_results: [{**_HIT, "title": "Ok"}]
    )
    wc._set_last_search_backend("ddgs")
    out = wc.search_web_results("ok query", max_retries=1)
    assert "unavailable" not in out.lower()
    assert "disabled" not in out.lower()
    assert "too short" not in out.lower()
    assert "Traceback" not in out


def test_unavailable_is_structured_and_unusable(monkeypatch):
    monkeypatch.setattr(wc, "_web_text_search", lambda *a, **k: [])
    out = wc.search_web_results("empty query here", max_retries=1)
    assert out.startswith("Web search unavailable:")
    assert "searxng" in out.lower()
    assert "Traceback" not in out
    assert web_search_body_usable(out) is False


def test_disabled_and_too_short(monkeypatch):
    monkeypatch.setattr(settings, "web_search_enabled", False, raising=False)
    assert "disabled" in wc.search_web_results("rust lang").lower()
    monkeypatch.setattr(settings, "web_search_enabled", True, raising=False)
    assert "too short" in wc.search_web_results("x").lower()
    assert web_search_body_usable(wc.search_web_results("x")) is False


def test_max_results_cap_passed_to_chain(monkeypatch):
    seen: list[int] = []

    def _search(q, max_results):
        seen.append(max_results)
        return [_HIT]

    monkeypatch.setattr(settings, "web_search_max_results", 99, raising=False)
    monkeypatch.setattr(wc, "_web_text_search", _search)
    wc.search_web_results("cap query", max_retries=1)
    assert seen == [10]  # hard cap in search_web_results


def test_tool_wrapper_calls_search_web_results(monkeypatch):
    monkeypatch.setattr(wt, "search_web_results", lambda q: f"WRAP:{q}")
    assert wt.search_web.invoke({"query": "ab"}) == "WRAP:ab"


# --- cache -------------------------------------------------------------------


def test_cache_hit_skips_all_backends(monkeypatch):
    sc.cache_put("cached query", "en", "searxng", [_HIT])
    order: list[str] = []
    _stub_chain(
        monkeypatch,
        order=order,
        searxng=AssertionError("HTTP must not run on cache hit"),
    )
    rows = wc._web_text_search("cached query", max_results=3)
    assert rows[0]["href"] == _HIT["href"]
    assert order == []
    assert wc.last_search_backend().startswith("cache:")


def test_wikipedia_results_are_not_cached(monkeypatch):
    _stub_chain(monkeypatch, wikipedia=[_HIT])
    assert wc._web_text_search("wiki only", max_results=2)
    assert wc.last_search_backend() == "wikipedia"
    # Second call must hit wikipedia again (not cache), because wiki is not stored.
    order: list[str] = []
    _stub_chain(monkeypatch, order=order, wikipedia=[_HIT])
    wc._web_text_search("wiki only", max_results=2)
    assert "wikipedia" in order
    assert not order or order[-1] == "wikipedia"


def test_searxng_results_are_cached_for_same_lang(monkeypatch):
    _stub_chain(monkeypatch, searxng=[_HIT])
    assert wc._web_text_search("store me", max_results=2)
    order: list[str] = []
    _stub_chain(
        monkeypatch,
        order=order,
        searxng=AssertionError("should be cache"),
    )
    rows = wc._web_text_search("store me", max_results=2)
    assert rows[0]["href"] == _HIT["href"]
    assert order == []
    assert wc.last_search_backend().startswith("cache:searxng")


def test_cache_ttl_expiry_refetches(monkeypatch):
    monkeypatch.setattr(sc.settings, "web_search_cache_ttl_sec", 10.0, raising=False)
    clock = {"t": 1_000.0}
    monkeypatch.setattr(sc.time, "time", lambda: clock["t"])
    _stub_chain(monkeypatch, searxng=[_HIT])
    wc._web_text_search("ttl query", max_results=2)
    clock["t"] = 1_020.0  # past TTL
    order: list[str] = []
    fresh = {**_HIT, "title": "Fresh"}
    _stub_chain(monkeypatch, order=order, searxng=[fresh])
    rows = wc._web_text_search("ttl query", max_results=2)
    assert order == ["searxng"]
    assert rows[0]["title"] == "Fresh"


# --- language / wikipedia ----------------------------------------------------


def test_search_lang_is_english_only():
    assert wc._search_lang("dollar rate") == "en"
    assert wc._search_lang("python docs") == "en"


def test_wikipedia_opensearch_parses_payload(monkeypatch):
    calls: list[str] = []

    class _Resp:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class _Client:
        def get(self, url, params=None, headers=None):
            calls.append(url)
            if "en.wikipedia.org" in url:
                return _Resp(
                    [
                        "q",
                        ["Python"],
                        ["A language"],
                        ["https://en.wikipedia.org/wiki/Python"],
                    ]
                )
            return _Resp(["q", [], [], []])

    monkeypatch.setattr(wc, "_search_http_client", lambda: _Client())
    rows = wc._wikipedia_opensearch_rows("Python", max_results=3)
    assert rows[0]["title"] == "Python"
    assert "wikipedia.org/wiki/Python" in rows[0]["href"]
    assert any("en.wikipedia.org" in u for u in calls)


# --- retries / backend label integrity --------------------------------------


def test_retries_until_rows(monkeypatch):
    monkeypatch.setattr(settings, "web_search_max_retries", 3, raising=False)
    n = {"i": 0}

    def _search(q, max_results):
        n["i"] += 1
        if n["i"] < 3:
            return []
        return [_HIT]

    monkeypatch.setattr(wc, "_web_text_search", _search)
    wc._set_last_search_backend("ddgs")
    out = wc.search_web_results("retry me please", max_retries=3)
    assert n["i"] == 3
    assert "Example Hit" in out
    assert web_search_body_usable(out) is True


def test_last_search_backend_is_thread_local(monkeypatch):
    """Live + solitary must not clobber each other's backend label."""
    barrier = threading.Barrier(2)
    labels: dict[str, str] = {}

    def _worker(name: str, backend: str):
        wc._set_last_search_backend(backend)
        barrier.wait()
        labels[name] = wc.last_search_backend()

    t1 = threading.Thread(target=_worker, args=("a", "searxng"))
    t2 = threading.Thread(target=_worker, args=("b", "ddgs"))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert labels["a"] == "searxng"
    assert labels["b"] == "ddgs"


def test_output_backend_label_matches_chain(monkeypatch):
    def _search(q, max_results):
        wc._set_last_search_backend("ddgs")
        return [_HIT]

    monkeypatch.setattr(wc, "_web_text_search", _search)
    out = wc.search_web_results("label query", max_retries=1)
    assert "(backend=ddgs)" in out


def test_live_searxng_smoke_if_up():
    """If local SearXNG is up, search_web must return usable hits (not unavailable)."""
    if not wc.check_searxng_health():
        pytest.skip("SearXNG not healthy on loopback")
    q = f"persistent-agent searxng smoke {int(time.time())}"
    out = wc.search_web_results(q, max_retries=1)
    assert web_search_body_usable(out), out[:400]
    assert "backend=" in out
    assert "unavailable" not in out.lower()
    # Prefer primary when healthy; cache: also OK on repeat within process.
    assert (
        "backend=searxng" in out
        or "backend=cache:searxng" in out
        or "backend=ddgs" in out
    )
