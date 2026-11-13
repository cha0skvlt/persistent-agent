"""Auto web search on user intent (before LLM tool-calling)."""

from __future__ import annotations

import re

from src.config import settings
from src.obs.logs import log_event
from src.tools.web_evidence import web_memory_usable, web_search_body_usable
from src.tools.web_intent import (
    resolve_page_url,
    suggest_web_search_query,
    wants_url_fetch,
)

WEB_PREFETCH_MARKER = "[WEB] search_web:"
WEB_FETCH_MARKER = "[WEB] fetch_url:"
_HTTP_STATUS_RE = re.compile(r"HTTP \d{3}\b")


def _prefetch_failed(text: str) -> bool:
    low = (text or "").lower()
    if "empty query" in low or "disabled" in low:
        return True
    if "web search failed" in low:
        return True
    if "fetch failed" in low:
        return True
    if _HTTP_STATUS_RE.search(text):
        return True
    if "url rejected" in low:
        return True
    if "unsupported content-type" in low:
        return True
    if "no readable text" in low:
        return True
    if "too large" in low:
        return True
    return False


def web_prefetch_succeeded(memories: list | None) -> bool:
    """True when retrieval prefetch returned usable live web evidence."""
    for mem in memories or []:
        if web_memory_usable(str(mem)):
            return True
    return False


def web_prefetch_truncated(memories: list | None) -> bool:
    """True when the SYS copy of the prefetch is a clipped view of the page.

    Memories hold the full body; ``_clip_evidence_body`` trims it to
    ``WEB_EVIDENCE_MAX_CHARS`` only when building the SYS block. So a body
    longer than that limit means the model is reading a prefix and the answer
    may sit past the cut — e.g. PEP 8 puts "79 characters" at offset ~6.9k
    while the evidence cap is 4.5k, leaving only the heading visible. Callers
    use this to keep the fetch tool bound so the full text stays reachable.
    """
    limit = int(settings.web_evidence_max_chars)
    if limit <= 0:
        return False
    for mem in memories or []:
        text = str(mem)
        if not text.startswith((WEB_PREFETCH_MARKER, WEB_FETCH_MARKER)):
            continue
        if len(text) > limit:
            return True
    return False


def prefetch_attempted_tool_name(memories: list | None) -> str | None:
    """Tool already invoked in retrieval prefetch (footer), usable or not."""
    for mem in memories or []:
        text = str(mem)
        if text.startswith(WEB_FETCH_MARKER):
            return "fetch_url"
        if text.startswith(WEB_PREFETCH_MARKER):
            return "search_web"
    return None


def prefetched_tool_name(memories: list | None) -> str | None:
    """Tool already run in retrieval with usable output (hide duplicate binding)."""
    for mem in memories or []:
        text = str(mem)
        if text.startswith(WEB_FETCH_MARKER) and web_memory_usable(text):
            return "fetch_url"
        if text.startswith(WEB_PREFETCH_MARKER) and web_memory_usable(text):
            return "search_web"
    return None


def prefetch_web_fetch(url: str) -> list[str]:
    """Run ``fetch_url`` immediately; return memories block for retrieval."""
    u = (url or "").strip()
    if not u:
        return [f"{WEB_FETCH_MARKER} empty url"]

    if not settings.web_fetch_enabled:
        return [f"{WEB_FETCH_MARKER} disabled (WEB_FETCH_ENABLED=0)"]

    from src.tools.web_core import fetch_url_text

    body = fetch_url_text(u)
    ok = not _prefetch_failed(body) and not body.startswith("HTTP ")
    log_event("web.prefetch_fetch", url=u, ok=ok)
    return [f"{WEB_FETCH_MARKER} url={u!r}\n{body}"]


def prefetch_web_search(user_message: str) -> list[str]:
    """Run ``search_web`` immediately; return memories block for retrieval."""
    q = suggest_web_search_query(user_message) or (user_message or "").strip()
    if not q:
        return [f"{WEB_PREFETCH_MARKER} empty query"]

    if not settings.web_search_enabled:
        return [f"{WEB_PREFETCH_MARKER} disabled (WEB_SEARCH_ENABLED=0)"]

    from src.tools.web_core import search_web_results

    body = search_web_results(q, max_retries=int(settings.web_search_max_retries))
    ok = web_search_body_usable(body)
    log_event("web.prefetch", query=q, ok=ok, usable=ok)
    return [f"{WEB_PREFETCH_MARKER} query={q!r}\n{body}"]


def prefetch_web_for_message(
    user_message: str, *, prior_human: list[str] | None = None
) -> list[str]:
    """Prefetch fetch when a URL is present; otherwise search."""
    if wants_url_fetch(user_message, prior_human):
        url = resolve_page_url(user_message, prior_human)
        if url:
            return prefetch_web_fetch(url)
    return prefetch_web_search(user_message)


def auto_tools_from_state(result: dict) -> list[str]:
    """Tools run outside ToolNode (e.g. retrieval prefetch) for turn footer."""
    mems = result.get("retrieved_memories") or []
    name = prefetch_attempted_tool_name(mems)
    return [name] if name else []
