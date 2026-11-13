"""Local Playwright browse session — open / follow / click / extract (no cloud).

All Playwright calls run on a dedicated worker thread. Sync Playwright is
greenlet-bound; LangGraph tool/agent nodes may hop threads, so a process-global
session without affinity leaks ``cannot switch to a different thread`` on close.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from concurrent.futures import Future
from typing import Any, Callable, TypeVar
from urllib.parse import urljoin, urlparse

from src.config import settings
from src.obs.logs import log_event

_log = logging.getLogger(__name__)
_T = TypeVar("_T")

_session: "_BrowseSession | None" = None
_worker_q: queue.Queue[tuple[Future[Any], Callable[[], Any]] | None] | None = None
_worker_thread: threading.Thread | None = None
_worker_lock = threading.Lock()
# Serialize tool entrypoints: one process-local browser; live + solitary must not interleave.
_api_lock = threading.Lock()


class _BrowseSession:
    def __init__(self) -> None:
        self.playwright: Any = None
        self.browser: Any = None
        self.context: Any = None
        self.page: Any = None
        self.steps = 0
        self.started = time.monotonic()
        self.current_url = ""

    def close(self) -> None:
        for attr in ("page", "context", "browser", "playwright"):
            obj = getattr(self, attr, None)
            if obj is None:
                continue
            try:
                if attr == "playwright":
                    obj.stop()
                else:
                    obj.close()
            except Exception as exc:
                _log.warning("browse close %s failed: %s", attr, exc)
            setattr(self, attr, None)
        self.current_url = ""
        self.steps = 0


def _ensure_worker() -> None:
    global _worker_q, _worker_thread
    with _worker_lock:
        if _worker_thread is not None and _worker_thread.is_alive():
            return
        _worker_q = queue.Queue()

        def _run() -> None:
            assert _worker_q is not None
            while True:
                item = _worker_q.get()
                if item is None:
                    break
                fut, fn = item
                try:
                    fut.set_result(fn())
                except BaseException as exc:  # noqa: BLE001 — forward to caller
                    fut.set_exception(exc)

        _worker_thread = threading.Thread(
            target=_run, name="agent-browse-pw", daemon=True
        )
        _worker_thread.start()


def _on_worker(fn: Callable[[], _T], *, timeout: float | None = None) -> _T:
    """Run ``fn`` on the Playwright worker thread and return its result."""
    _ensure_worker()
    assert _worker_q is not None
    fut: Future[_T] = Future()
    _worker_q.put((fut, fn))
    wait = timeout
    if wait is None:
        wait = max(30.0, float(settings.web_browse_timeout_sec) * 3.0)
    return fut.result(timeout=wait)


def _close_unlocked() -> None:
    global _session
    if _session is not None:
        _session.close()
        _session = None


def _close_on_worker(*, timeout: float = 15.0) -> None:
    def _close() -> None:
        _close_unlocked()

    try:
        _on_worker(_close, timeout=timeout)
    except Exception as exc:
        _log.warning("browse session close failed: %s", exc)
        global _session
        _session = None


def close_browse_session() -> None:
    """Force-close the process-local browse session (tests / turn end / fail-closed)."""
    with _api_lock:
        _close_on_worker()


def _ensure_session() -> _BrowseSession:
    global _session
    if _session is not None and _session.browser is not None:
        return _session
    _close_unlocked()
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError("playwright not installed") from exc
    sess = _BrowseSession()
    try:
        sess.playwright = sync_playwright().start()
        sess.browser = sess.playwright.chromium.launch(headless=True)
        timeout_ms = int(float(settings.web_browse_timeout_sec) * 1000)
        # Service workers can fetch outside the page's request pipeline.
        sess.context = sess.browser.new_context(service_workers="block")
        sess.page = sess.context.new_page()
        sess.page.set_default_timeout(timeout_ms)
        _install_request_guard(sess.page)
    except Exception as exc:
        sess.close()
        msg = str(exc)
        if "Executable doesn't exist" in msg or "playwright install" in msg.lower():
            raise RuntimeError(
                "Playwright Chromium not installed — run: "
                "venv/bin/python -m playwright install chromium"
            ) from exc
        raise
    _session = sess
    return sess


def _step_guard(sess: _BrowseSession) -> str | None:
    max_steps = max(1, int(settings.web_browse_max_steps))
    if sess.steps >= max_steps:
        return (
            f"browse_error: step cap reached ({max_steps}). "
            "Call action=close or start a new open."
        )
    return None


def _assert_public_url(url: str) -> str:
    """SSRF gate — same rules as fetch_url (public http(s) only)."""
    from src.tools.web_core import normalize_public_http_url

    return normalize_public_http_url(url)


# Schemes that never leave the browser process. Everything else must pass the gate.
_SAFE_SCHEMES = ("data:", "blob:", "about:", "chrome-extension:", "devtools:")


def _install_request_guard(page: Any) -> None:
    """Block every browser request to a non-public target, not just the top page.

    ``_safe_goto`` validates the navigation URL, but a page can still pull
    subresources from ``127.0.0.1``, ``169.254.169.254``, or a private range via
    an iframe, an image, a script, or an XHR. This guard runs on every request,
    including redirects and post-click navigation.
    """

    def _guard(route: Any) -> None:
        try:
            url = route.request.url or ""
        except Exception:
            try:
                route.abort()
            except Exception:
                pass
            return
        if url.startswith(_SAFE_SCHEMES):
            try:
                route.continue_()
            except Exception:
                pass
            return
        try:
            _assert_public_url(url)
        except Exception:
            try:
                route.abort()
            except Exception:
                pass
            return
        try:
            route.continue_()
        except Exception:
            pass

    page.route("**/*", _guard)


def _safe_goto(sess: _BrowseSession, url: str) -> str:
    safe = _assert_public_url(url)
    assert sess.page is not None
    timeout_ms = int(float(settings.web_browse_timeout_sec) * 1000)
    sess.page.goto(safe, wait_until="domcontentloaded")
    # Short settle for late-rendered DOM (any site). Cap so long-poll pages don't hang.
    try:
        sess.page.wait_for_load_state(
            "networkidle",
            timeout=min(3000, timeout_ms),
        )
    except Exception:
        pass
    final = (sess.page.url or safe).strip()
    try:
        final_safe = _assert_public_url(final)
    except ValueError as exc:
        raise RuntimeError(
            f"navigation landed on blocked URL: {final} ({exc})"
        ) from exc
    if urlparse(final_safe).scheme not in {"http", "https"}:
        raise RuntimeError(f"navigation landed on blocked scheme: {final_safe}")
    sess.current_url = final_safe
    return sess.current_url


def _visible_page_text(sess: _BrowseSession, *, max_chars: int) -> str:
    """Browser-rendered full page text (body.innerText), HTML fallback."""
    from src.tools.web_extract import compact_visible_text, extract_page_text

    assert sess.page is not None
    text = ""
    try:
        raw = sess.page.inner_text("body")
        if isinstance(raw, str) and raw.strip():
            text = compact_visible_text(raw)
    except Exception:
        text = ""
    if not text:
        html = sess.page.content()
        text = extract_page_text(html, max_chars=max_chars)
    if max_chars > 0 and len(text) > max_chars:
        return text[:max_chars] + "…"
    return text


def _extract_payload(sess: _BrowseSession, *, requested_url: str = "") -> str:
    from src.tools.web_extract import redirect_note

    assert sess.page is not None
    cur = (sess.page.url or sess.current_url or "").strip()
    try:
        cur = _assert_public_url(cur)
    except ValueError as exc:
        raise RuntimeError(f"page URL blocked: {cur} ({exc})") from exc
    sess.current_url = cur
    title = ""
    try:
        title = (sess.page.title() or "").strip()
    except Exception:
        title = ""
    head = f"URL: {cur}"
    note = redirect_note(requested_url, cur)
    if note:
        head += f"\n{note}"
    if title:
        head += f"\nTitle: {title}"
    text = _visible_page_text(sess, max_chars=int(settings.web_browse_max_chars))
    if not text:
        return f"{head}\n(no readable text extracted)"
    return f"{head}\n---\n{text}"


def _browse_web_on_worker(action: str, target: str) -> str:
    act = (action or "").strip().lower()
    tgt = (target or "").strip()

    if act == "close":
        _close_unlocked()
        log_event("web.browse", action="close", ok=True)
        return "browse_closed"

    if act == "open":
        if not tgt:
            return "browse_error: open requires target=URL"
        _close_unlocked()
        sess = _ensure_session()
        guard = _step_guard(sess)
        if guard:
            return guard
        sess.steps += 1
        url = _safe_goto(sess, tgt)
        log_event("web.browse", action="open", url=url, steps=sess.steps)
        return _extract_payload(sess, requested_url=tgt)

    sess = _session
    if sess is None or sess.page is None:
        return "browse_error: no session — call action=open first"

    guard = _step_guard(sess)
    if guard:
        _close_unlocked()
        return guard

    if act == "follow":
        if not tgt:
            return "browse_error: follow requires target=URL or path"
        if tgt.isdigit():
            return "browse_error: follow needs a URL or path, not a link index"
        sess.steps += 1
        base = sess.current_url or sess.page.url or ""
        requested = urljoin(base, tgt)
        url = _safe_goto(sess, requested)
        log_event("web.browse", action="follow", url=url, steps=sess.steps)
        return _extract_payload(sess, requested_url=requested)

    if act == "click":
        if not tgt:
            return "browse_error: click requires target=CSS selector or link text"
        sess.steps += 1
        timeout_ms = int(float(settings.web_browse_timeout_sec) * 1000)
        try:
            sess.page.click(tgt, timeout=timeout_ms)
        except Exception:
            sess.page.get_by_text(tgt, exact=False).first.click(timeout=timeout_ms)
        sess.page.wait_for_load_state("domcontentloaded")
        final = (sess.page.url or sess.current_url or "").strip()
        try:
            sess.current_url = _assert_public_url(final)
        except ValueError as exc:
            _close_unlocked()
            return f"browse_error: click landed on blocked URL: {final} ({exc})"
        log_event(
            "web.browse",
            action="click",
            url=sess.current_url,
            steps=sess.steps,
        )
        return _extract_payload(sess)

    if act == "extract":
        sess.steps += 1
        log_event(
            "web.browse", action="extract", url=sess.current_url, steps=sess.steps
        )
        return _extract_payload(sess)

    return "browse_error: unknown action"


def browse_web_impl(
    action: str = "",
    target: str = "",
) -> str:
    """Run one browse action. Fail-closed: errors as strings, session cleaned on crash."""
    if not bool(getattr(settings, "web_browse_enabled", True)):
        return "browse_error: disabled (WEB_BROWSE_ENABLED=0)"
    act = (action or "").strip().lower()
    tgt = (target or "").strip()
    if act not in {"open", "follow", "click", "extract", "close"}:
        return (
            "browse_error: action must be open|follow|click|extract|close. "
            "open/follow need target=URL (or relative path for follow); "
            "click needs target=CSS or link text."
        )

    with _api_lock:
        try:
            return _on_worker(lambda: _browse_web_on_worker(act, tgt))
        except Exception as exc:
            _log.warning("browse_web failed action=%s err=%s", act, exc)
            try:
                _close_on_worker()
            except Exception:
                pass
            log_event(
                "web.browse",
                action=act,
                ok=False,
                err=type(exc).__name__,
            )
            return f"browse_error: {type(exc).__name__}: {exc}"


__all__ = [
    "browse_web_impl",
    "close_browse_session",
]
