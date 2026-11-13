"""HTTP fetch + local search chain (SearXNG primary; keyless fallbacks)."""

from __future__ import annotations

import ipaddress
import logging
import re
import socket
import threading
import time
from html import unescape
from typing import Any
from urllib.parse import quote_plus, urljoin, urlparse, urlunparse

import httpx

from src.config import settings
from src.obs.logs import log_event

_log = logging.getLogger(__name__)

_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_SCRIPT_STYLE_RE = re.compile(
    r"<(script|style|noscript)[^>]*>.*?</\1>",
    re.IGNORECASE | re.DOTALL,
)
_TAG_RE = re.compile(r"<[^>]+>")
_REDIRECT_STATUS = frozenset({301, 302, 303, 307, 308})
_MAX_REDIRECTS = 5
_DEFAULT_SEARCH_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)
_thread_local = threading.local()
_search_thread_local = threading.local()
_CSS_JUNK_RE = re.compile(r"\.css-[\w-]+\{[^}]*\}|@media[^{]*\{[^}]*\}")
_DNS_PUBLIC_CACHE_TTL_SEC = 60.0
_dns_public_cache: dict[str, tuple[bool, float]] = {}
_dns_cache_lock = threading.Lock()
_searxng_warned_down: bool = False

# ── Search backend circuit breaker ────────────────────────────────────────────
# A captcha/timeout streak on one backend must not stall every live turn while
# the chain grinds to Wikipedia — the turn holds the LLM traffic-light
# semaphore the whole time. Consecutive failures open the breaker for a
# cooldown; the chain skips open backends instantly.
_breaker_lock = threading.Lock()
_breaker_fails: dict[str, int] = {}
_breaker_open_until: dict[str, float] = {}


def _breaker_is_open(name: str) -> bool:
    with _breaker_lock:
        until = _breaker_open_until.get(name, 0.0)
    if until <= 0.0:
        return False
    if time.monotonic() >= until:
        with _breaker_lock:
            _breaker_open_until.pop(name, None)
            _breaker_fails.pop(name, None)
        return False
    return True


def _breaker_record(name: str, *, ok: bool) -> None:
    threshold = max(1, int(settings.web_backend_fail_threshold))
    cooldown = max(1.0, float(settings.web_backend_cooldown_sec))
    with _breaker_lock:
        if ok:
            _breaker_fails.pop(name, None)
            _breaker_open_until.pop(name, None)
            return
        fails = _breaker_fails.get(name, 0) + 1
        _breaker_fails[name] = fails
        if fails >= threshold:
            _breaker_open_until[name] = time.monotonic() + cooldown
    if fails >= threshold:
        log_event(
            "web.backend_breaker_open",
            backend=name,
            fails=fails,
            cooldown_sec=cooldown,
        )


def reset_search_breaker_for_tests() -> None:
    with _breaker_lock:
        _breaker_fails.clear()
        _breaker_open_until.clear()


def _set_last_search_backend(name: str) -> None:
    """Record which backend answered — thread-local: live and background turns
    run concurrently and must not clobber each other's label."""
    _search_thread_local.backend = name


def last_search_backend() -> str:
    return str(getattr(_search_thread_local, "backend", "") or "")


def _user_agent() -> str:
    return (settings.web_user_agent or "PersistentAgent-Agent/1.0").strip()


def _http_timeout() -> httpx.Timeout:
    read = float(settings.web_fetch_timeout_sec)
    return httpx.Timeout(connect=5.0, read=read, write=read, pool=5.0)


def _http_client() -> httpx.Client:
    client = getattr(_thread_local, "client", None)
    if client is None or client.is_closed:
        client = httpx.Client(
            timeout=_http_timeout(),
            headers={"User-Agent": _user_agent()},
            follow_redirects=False,
        )
        _thread_local.client = client
    return client


def reset_http_client() -> None:
    """Close thread-local client (tests)."""
    client = getattr(_thread_local, "client", None)
    if client is not None and not client.is_closed:
        client.close()
    _thread_local.client = None
    reset_search_http_client()


def _search_user_agent() -> str:
    custom = getattr(settings, "web_search_user_agent", None)
    if custom and str(custom).strip():
        return str(custom).strip()
    return _DEFAULT_SEARCH_UA


def _search_http_client() -> httpx.Client:
    client = getattr(_search_thread_local, "client", None)
    if client is None or client.is_closed:
        client = httpx.Client(
            timeout=_http_timeout(),
            headers={"User-Agent": _search_user_agent()},
            follow_redirects=True,
        )
        _search_thread_local.client = client
    return client


def reset_search_http_client() -> None:
    """Close thread-local search client (tests)."""
    client = getattr(_search_thread_local, "client", None)
    if client is not None and not client.is_closed:
        client.close()
    _search_thread_local.client = None


def _clean_search_title(title: str) -> str:
    t = _CSS_JUNK_RE.sub("", title or "")
    t = re.sub(r"\s+", " ", t).strip()
    return t or "(no title)"


def _blocked_host(host: str) -> bool:
    h = (host or "").strip().lower().rstrip(".")
    if not h:
        return True
    if h in {"localhost", "metadata.google.internal", "metadata"}:
        return True
    if h.endswith(".local") or h.endswith(".internal") or h.endswith(".localhost"):
        return True
    try:
        ip = ipaddress.ip_address(h)
        return (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
        )
    except ValueError:
        return False


def _ip_is_public(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    # Fail closed: CGNAT 100.64/10 is not is_private on some Pythons but
    # is_global is False — require global unicast only.
    return bool(getattr(ip, "is_global", False))


def clear_dns_public_cache() -> None:
    """Clear SSRF DNS cache (tests / after network policy changes)."""
    with _dns_cache_lock:
        _dns_public_cache.clear()


def _resolve_public_ips_uncached(host: str) -> list[str]:
    """Resolve host to public IPs only; fail closed if any address is non-public."""
    if _blocked_host(host):
        return []
    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except OSError:
        return []
    if not infos:
        return []
    ips: list[str] = []
    for info in infos:
        addr = info[4][0]
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            continue
        if not _ip_is_public(ip):
            return []
        ips.append(addr)
    return ips


def _resolve_host_is_public_uncached(host: str) -> bool:
    return bool(_resolve_public_ips_uncached(host))


def _resolve_host_is_public(host: str) -> bool:
    key = (host or "").strip().lower().rstrip(".")
    if not key:
        return False
    now = time.monotonic()
    with _dns_cache_lock:
        cached = _dns_public_cache.get(key)
        if cached is not None and now - cached[1] < _DNS_PUBLIC_CACHE_TTL_SEC:
            return cached[0]
    result = _resolve_host_is_public_uncached(key)
    with _dns_cache_lock:
        _dns_public_cache[key] = (result, now)
    return result


def normalize_public_http_url(url: str) -> str:
    """Validate URL scheme/host; block private/metadata targets (SSRF)."""
    raw = (url or "").strip()
    if not raw:
        raise ValueError("URL is empty")
    if raw.startswith("//"):
        raw = "https:" + raw
    parsed = urlparse(raw)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only http/https URLs are allowed")
    host = parsed.hostname
    if not host or _blocked_host(host):
        raise ValueError("URL host is not allowed")
    if not _resolve_host_is_public(host):
        raise ValueError("URL host resolves to a private or local address")
    port = parsed.port
    if port is not None and port not in {80, 443, 8080, 8443}:
        raise ValueError("URL port is not allowed")
    clean = parsed._replace(fragment="")
    return urlunparse(clean)


def html_to_text(html: str, *, max_chars: int | None = None) -> str:
    """Full visible page text from HTML."""
    from src.tools.web_extract import extract_readable_text

    return extract_readable_text(html, max_chars=max_chars)


def extract_title(html: str) -> str:
    m = _TITLE_RE.search(html or "")
    if not m:
        return ""
    return unescape(re.sub(r"\s+", " ", m.group(1))).strip()


def _read_limited_response(resp: httpx.Response, *, max_bytes: int) -> bytes:
    chunks: list[bytes] = []
    size = 0
    for chunk in resp.iter_bytes():
        if not chunk:
            continue
        size += len(chunk)
        if size > max_bytes:
            raise ValueError(f"response exceeds {max_bytes} bytes")
        chunks.append(chunk)
    return b"".join(chunks)


def _pinned_fetch_targets(url: str) -> tuple[str, dict[str, str]]:
    """Map a validated URL to an IP-literal request URL + Host header (DNS rebinding guard)."""
    parsed = urlparse(url)
    host = parsed.hostname
    if not host:
        raise ValueError("URL host is not allowed")
    ips = _resolve_public_ips_uncached(host)
    if not ips:
        raise ValueError("URL host resolves to a private or local address")
    ip = ips[0]
    ip_obj = ipaddress.ip_address(ip)
    host_part = f"[{ip}]" if ip_obj.version == 6 else ip
    port = parsed.port
    if port is None:
        port = 443 if parsed.scheme == "https" else 80
    default_port = 443 if parsed.scheme == "https" else 80
    if port != default_port:
        host_part = f"{host_part}:{port}"
    pinned = urlunparse(parsed._replace(netloc=host_part))
    return pinned, {"Host": host}


def _safe_http_get(url: str) -> tuple[str, dict[str, str], bytes]:
    """GET with per-hop SSRF validation and DNS-pinned connect (no blind redirect follow)."""
    client = _http_client()
    current = normalize_public_http_url(url)
    max_bytes = int(settings.web_fetch_max_bytes)

    for hop in range(_MAX_REDIRECTS + 1):
        pinned_url, pin_headers = _pinned_fetch_targets(current)
        req_headers = {
            "Accept": "text/html,application/xhtml+xml",
            **pin_headers,
        }
        stream_kwargs: dict = {"headers": req_headers}
        host_header = pin_headers.get("Host") or ""
        if host_header:
            stream_kwargs["extensions"] = {"sni_hostname": host_header}
        with client.stream(
            "GET",
            pinned_url,
            **stream_kwargs,
        ) as resp:
            if resp.status_code in _REDIRECT_STATUS:
                location = (resp.headers.get("location") or "").strip()
                if not location:
                    resp.raise_for_status()
                if hop >= _MAX_REDIRECTS:
                    raise ValueError("too many redirects")
                nxt = normalize_public_http_url(urljoin(current, location))
                if (
                    urlparse(current).scheme == "https"
                    and urlparse(nxt).scheme == "http"
                ):
                    raise ValueError("redirect downgrades https to http")
                current = nxt
                continue
            resp.raise_for_status()
            raw = _read_limited_response(resp, max_bytes=max_bytes)
            headers = {k.lower(): v for k, v in resp.headers.items()}
            return current, headers, raw

    raise ValueError("too many redirects")  # pragma: no cover


# Anti-bot / challenge statuses where a real browser render may still succeed.
_RENDER_FALLBACK_STATUS = frozenset({401, 403, 406, 407, 429, 503})


def _render_fallback_enabled() -> bool:
    return bool(getattr(settings, "web_fetch_render_fallback", True)) and bool(
        getattr(settings, "web_browse_enabled", True)
    )


def _render_page_html(url: str) -> tuple[str, str]:
    """One-shot local Playwright render. Returns ``(final_url, html)``; raises on failure."""
    from playwright.sync_api import sync_playwright

    safe = normalize_public_http_url(url)
    timeout_ms = int(float(settings.web_browse_timeout_sec) * 1000)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            page.set_default_timeout(timeout_ms)
            page.goto(safe, wait_until="domcontentloaded")
            # Give SPA shells one beat to hydrate before reading the DOM.
            try:
                page.wait_for_load_state("networkidle", timeout=min(timeout_ms, 5000))
            except Exception:
                pass
            final = normalize_public_http_url((page.url or safe).strip())
            html = page.content() or ""
        finally:
            browser.close()
    return final, html


def _rendered_or(static_result: str, url: str, *, static_text_len: int = 0) -> str:
    """Try the JS render; keep the static outcome when render fails or adds nothing."""
    if not _render_fallback_enabled():
        return static_result
    try:
        final_url, html = _render_page_html(url)
    except Exception as exc:
        _log.warning("web.fetch render fallback failed url=%s err=%s", url, exc)
        log_event("web.fetch_render_failed", url=url, err=type(exc).__name__)
        return static_result
    from src.tools.web_extract import extract_page_text, redirect_note

    text = extract_page_text(html)
    if len(text) <= static_text_len:
        return static_result
    head = f"URL: {final_url}"
    note = redirect_note(url, final_url)
    if note:
        head += f"\n{note}"
    title = extract_title(html)
    if title:
        head += f"\nTitle: {title}"
    head += "\n(rendered)"
    log_event("web.fetch_rendered", url=final_url, chars=len(text))
    return f"{head}\n---\n{text}"


def fetch_url_text(url: str) -> str:
    """Return one page whole: static GET first, local JS-render fallback.

    Escalates to a one-shot Playwright render when the static fetch is blocked
    (anti-bot status) or the extracted text is too thin (JS-only page), so one
    call returns the full page either way.
    """
    if not settings.web_fetch_enabled:
        return "Web fetch is disabled (WEB_FETCH_ENABLED=0)."
    try:
        safe_url, headers, raw = _safe_http_get(url)
    except ValueError as exc:
        msg = str(exc)
        if "bytes" in msg:
            return f"URL response too large (>{settings.web_fetch_max_bytes} bytes)"
        return f"URL rejected: {exc}"
    except httpx.HTTPStatusError as exc:
        code = exc.response.status_code if exc.response is not None else 0
        static_err = f"HTTP {code} for {url}"
        if code in _RENDER_FALLBACK_STATUS:
            return _rendered_or(static_err, url)
        return static_err
    except Exception as exc:
        _log.warning("web.fetch failed url=%s err=%s", url, exc)
        return f"Fetch failed: {exc}"

    ctype = (headers.get("content-type") or "").lower()
    ctype_base = ctype.split(";", 1)[0].strip()
    allowed_ct = (
        "text/html",
        "application/xhtml",
        "text/plain",
        "text/markdown",
        "application/json",
    )
    if not ctype_base or not any(token in ctype_base for token in allowed_ct):
        return (
            f"URL: {safe_url}\n"
            f"(unsupported content-type: {ctype_base or 'missing'})"
        )
    charset = "utf-8"
    if "charset=" in ctype:
        charset = ctype.split("charset=", 1)[-1].split(";")[0].strip() or charset
    try:
        html = raw.decode(charset, errors="replace")
    except LookupError:
        html = raw.decode("utf-8", errors="replace")

    from src.tools.web_extract import extract_page_text, redirect_note

    head = f"URL: {safe_url}"
    note = redirect_note(url, safe_url)
    if note:
        head += f"\n{note}"

    # Non-HTML: return decoded body as-is (do not run the HTML page extractor).
    if "html" not in ctype_base:
        limit = int(settings.web_fetch_max_chars)
        text = html.strip()
        if limit > 0 and len(text) > limit:
            text = text[:limit] + "…"
        if not text:
            return f"{head}\n(no readable text extracted)"
        return f"{head}\n---\n{text}"

    title = extract_title(html)
    if title:
        head += f"\nTitle: {title}"
    text = extract_page_text(html)
    if not text:
        return _rendered_or(f"{head}\n(no readable text extracted)", url)
    min_chars = max(0, int(getattr(settings, "web_fetch_render_min_chars", 200)))
    if len(text) < min_chars:
        # JS shell («Loading…», bare menu) — render for the real page body.
        return _rendered_or(f"{head}\n---\n{text}", url, static_text_len=len(text))
    return f"{head}\n---\n{text}"


def _search_lang(query: str) -> str:
    _ = query
    return "en"


def _searxng_base_url() -> str:
    return (getattr(settings, "searxng_url", None) or "http://127.0.0.1:8888").rstrip(
        "/"
    )


def _assert_loopback_searxng(base: str) -> str:
    """SearXNG is loopback-only — never call a remote aggregator as primary."""
    parsed = urlparse(base)
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in {"http", "https"} or host not in {"127.0.0.1", "localhost"}:
        raise ValueError(f"SEARXNG_URL must be loopback, got host={host!r}")
    return base


def check_searxng_health() -> bool:
    """One-shot health probe for API/Heart boot. WARNING on failure; never raises."""
    global _searxng_warned_down
    base = _searxng_base_url()
    try:
        base = _assert_loopback_searxng(base)
        timeout = float(getattr(settings, "searxng_timeout_sec", 10.0) or 10.0)
        with httpx.Client(timeout=min(5.0, timeout)) as client:
            resp = client.get(
                f"{base}/search",
                params={"q": "healthcheck", "format": "json"},
            )
            if resp.status_code != 200:
                raise ValueError(f"status={resp.status_code}")
            data = resp.json()
            if not isinstance(data, dict):
                raise ValueError("non-json body")
        if _searxng_warned_down:
            _log.info("searxng.health recovered url=%s", base)
        _searxng_warned_down = False
        log_event("searxng.health_ok", url=base)
        return True
    except Exception as exc:
        _searxng_warned_down = True
        _log.warning("searxng.unavailable url=%s err=%s", base, exc)
        log_event("searxng.health_fail", url=base, err=str(exc)[:200])
        return False


def _searxng_rows(query: str, *, max_results: int, lang: str) -> list[dict[str, Any]]:
    base = _assert_loopback_searxng(_searxng_base_url())
    timeout = float(getattr(settings, "searxng_timeout_sec", 10.0) or 10.0)
    with httpx.Client(timeout=timeout) as client:
        resp = client.get(
            f"{base}/search",
            params={"q": query, "format": "json", "language": lang},
        )
        if resp.status_code in {202, 429}:
            raise ValueError(f"search rate-limited ({resp.status_code})")
        if resp.status_code == 403:
            raise ValueError("searxng json format forbidden (enable search.formats)")
        resp.raise_for_status()
        data = resp.json()
    results: list[dict[str, Any]] = []
    for item in data.get("results") or []:
        if not isinstance(item, dict):
            continue
        href = str(item.get("url") or "").strip()
        title = str(item.get("title") or "").strip()
        body = str(item.get("content") or item.get("snippet") or "").strip()
        engine = str(item.get("engine") or "").strip()
        if not href:
            continue
        row: dict[str, Any] = {"title": title or href, "href": href, "body": body}
        if engine:
            row["engine"] = engine
        results.append(row)
        if len(results) >= max_results:
            break
    return results


def _ddgs_rows(query: str, *, max_results: int, lang: str) -> list[dict[str, Any]]:
    from ddgs import DDGS

    region = (
        getattr(settings, "web_search_region", None) or "wt-wt"
    ).strip() or "wt-wt"
    if lang == "ru" and region == "wt-wt":
        region = "ru-ru"
    rows: list[dict[str, Any]] = []
    with DDGS() as client:
        hits = client.text(query, region=region, max_results=max_results) or []
        for item in hits:
            if not isinstance(item, dict):
                continue
            href = str(
                item.get("href") or item.get("link") or item.get("url") or ""
            ).strip()
            title = str(item.get("title") or "").strip()
            body = str(item.get("body") or item.get("snippet") or "").strip()
            if not href:
                continue
            rows.append({"title": title or href, "href": href, "body": body})
            if len(rows) >= max_results:
                break
    return rows


def _playwright_ddg_serp_rows(query: str, *, max_results: int) -> list[dict[str, Any]]:
    """Render DDG HTML SERP via local Playwright; extract results from DOM."""
    from playwright.sync_api import sync_playwright

    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    timeout_ms = int(float(settings.web_browse_timeout_sec) * 1000)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            page.set_default_timeout(timeout_ms)
            page.set_extra_http_headers({"User-Agent": _search_user_agent()})
            page.goto(url, wait_until="domcontentloaded")
            body_l = (page.content() or "").lower()
            if "captcha" in body_l[:8000] or "unusual traffic" in body_l[:8000]:
                raise ValueError("search captcha")
            raw_rows = page.eval_on_selector_all(
                "div.result, div.web-result, div.results_links_deep",
                """(nodes) => nodes.map((n) => {
                    const a = n.querySelector('a.result__a, a.result-link, h2 a, a[href]');
                    if (!a) return null;
                    const href = a.href || '';
                    const title = (a.textContent || '').trim();
                    const sn = n.querySelector('.result__snippet, .result-snippet, a.result__snippet');
                    const body = sn ? (sn.textContent || '').trim() : '';
                    return {href, title, body};
                }).filter(Boolean)""",
            )
        finally:
            browser.close()
    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw_rows or []:
        if not isinstance(item, dict):
            continue
        href = str(item.get("href") or "").strip()
        if not href or href in seen or "duckduckgo.com/y.js" in href:
            continue
        seen.add(href)
        results.append(
            {
                "title": str(item.get("title") or href).strip() or href,
                "href": href,
                "body": str(item.get("body") or "").strip(),
            }
        )
        if len(results) >= max_results:
            break
    return results


def _wikipedia_opensearch_rows(query: str, *, max_results: int) -> list[dict[str, Any]]:
    """MediaWiki opensearch — last-resort fallback."""
    client = _search_http_client()
    ordered = [("en.wikipedia.org", "https://en.wikipedia.org/wiki/")]
    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    for host, wiki_base in ordered:
        if len(results) >= max_results:
            break
        resp = client.get(
            f"https://{host}/w/api.php",
            params={
                "action": "opensearch",
                "search": query,
                "limit": str(max_results),
                "namespace": "0",
                "format": "json",
            },
            headers={"Accept": "application/json"},
        )
        resp.raise_for_status()
        payload = resp.json()
        if not isinstance(payload, list) or len(payload) < 4:
            continue
        titles = payload[1] if isinstance(payload[1], list) else []
        descs = payload[2] if isinstance(payload[2], list) else []
        urls = payload[3] if isinstance(payload[3], list) else []
        for i, title in enumerate(titles):
            if len(results) >= max_results:
                break
            href = str(urls[i] if i < len(urls) else "").strip()
            if not href:
                from urllib.parse import quote

                href = wiki_base + quote(str(title).replace(" ", "_"))
            if href in seen:
                continue
            seen.add(href)
            body = str(descs[i] if i < len(descs) else "").strip()
            results.append(
                {
                    "title": str(title or "(no title)"),
                    "href": href,
                    "body": body,
                }
            )
    return results


def _web_text_search(query: str, *, max_results: int) -> list[dict[str, Any]]:
    """Chain: searxng → ddgs → playwright SERP → wikipedia. Empty → []."""
    global _searxng_warned_down
    lang = _search_lang(query)
    from src.tools.search_cache import cache_get, cache_put

    cached = cache_get(query, lang)
    if cached is not None:
        rows, backend = cached
        _set_last_search_backend(f"cache:{backend}")
        return rows

    backends: list[tuple[str, Any]] = [
        ("searxng", lambda: _searxng_rows(query, max_results=max_results, lang=lang)),
        ("ddgs", lambda: _ddgs_rows(query, max_results=max_results, lang=lang)),
        (
            "playwright_serp",
            lambda: _playwright_ddg_serp_rows(query, max_results=max_results),
        ),
        (
            "wikipedia",
            lambda: _wikipedia_opensearch_rows(query, max_results=max_results),
        ),
    ]
    deadline = time.monotonic() + max(1.0, float(settings.web_search_total_budget_sec))
    for name, fn in backends:
        if time.monotonic() >= deadline:
            log_event("web.search_budget_exhausted", query=query, next_backend=name)
            _log.warning("web.search budget exhausted before backend=%s", name)
            break
        if _breaker_is_open(name):
            log_event("web.search_backend_skip", backend=name, reason="breaker_open")
            continue
        try:
            rows = fn()
            if rows:
                _breaker_record(name, ok=True)
                _set_last_search_backend(name)
                if name == "searxng":
                    _searxng_warned_down = False
                log_event("web.search_hit", backend=name, query=query, n=len(rows))
                _log.info(
                    "web.search_hit backend=%s query=%r n=%s", name, query, len(rows)
                )
                # Wikipedia is a last-resort stub: caching it would pin weak
                # results for the full TTL after the SERP backends recover.
                if name != "wikipedia":
                    cache_put(query, lang, name, rows)
                return rows
            # Empty result is an answer, not a backend failure.
            _breaker_record(name, ok=True)
            _log.info("web.search empty backend=%s query=%r", name, query)
        except Exception as exc:
            _breaker_record(name, ok=False)
            err_s = str(exc).lower()
            if name == "searxng":
                _searxng_warned_down = True
                _log.warning("searxng.unavailable query=%r err=%s", query, exc)
            else:
                _log.warning("web.search backend=%s query=%r err=%s", name, query, exc)
            if "rate-limited" in err_s or "captcha" in err_s:
                log_event("web.search_backend_skip", backend=name, reason=err_s[:80])
                continue
    _set_last_search_backend("")
    log_event("web.search_empty", query=query)
    return []


def search_web_results(query: str, *, max_retries: int | None = None) -> str:
    """Web text search; returns numbered snippets for the LLM."""
    if not settings.web_search_enabled:
        return "Web search is disabled (WEB_SEARCH_ENABLED=0)."
    from src.tools.web_intent import suggest_web_search_query

    q = suggest_web_search_query(query) if query else ""
    q = (q or (query or "").strip()).strip()
    if len(q) < 2:
        return "Search query too short."
    if len(q) > 500:
        q = q[:500]

    max_results = max(1, min(int(settings.web_search_max_results), 10))
    if max_retries is None:
        retries = max(1, min(int(settings.web_search_max_retries), 3))
    else:
        retries = max(1, min(int(max_retries), 5))
    rows: list[dict[str, Any]] = []

    for attempt in range(retries):
        try:
            rows = _web_text_search(q, max_results=max_results)
            if rows:
                break
        except Exception as exc:
            _log.warning(
                "web.search attempt=%s query=%r err=%s",
                attempt + 1,
                q,
                exc,
            )
        if attempt + 1 < retries:
            time.sleep(min(2.0, 0.4 * (attempt + 1)))

    if not rows:
        log_event("web.search_unavailable", query=q)
        return (
            "Web search unavailable: all backends failed "
            f"(searxng/ddgs/playwright/wikipedia) for: {q}"
        )

    backend = last_search_backend().strip() or "unknown"
    parts: list[str] = [f"Web search: {q} (backend={backend})", ""]
    for i, row in enumerate(rows, start=1):
        title = _clean_search_title(str(row.get("title") or "").strip())
        href = str(row.get("href") or row.get("link") or "").strip()
        body = str(row.get("body") or row.get("snippet") or "").strip()
        engine = str(row.get("engine") or "").strip()
        parts.append(f"{i}. {title}")
        if href:
            parts.append(f"   URL: {href}")
        if engine:
            parts.append(f"   engine: {engine}")
        if body:
            parts.append(f"   {body}")
        parts.append("")
    parts.append(
        "Use fetch_url or browse_web open on a result URL for the full page text. "
        "Answer ONLY with numbers and dates that appear in the snippets above; "
        "if the rate or fact is missing, say so — do not guess from memory. "
        "save_memory for facts worth keeping."
    )
    return "\n".join(parts).strip()
