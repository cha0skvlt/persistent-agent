"""Detect when the user asks for live web search / URL fetch."""

from __future__ import annotations

import re
from urllib.parse import urljoin, urlparse

from langchain_core.messages import BaseMessage, HumanMessage

_WEB_INTENT_RE = re.compile(
    r"(?:"
    r"search\s+(?:the\s+)?(?:web|internet|online)|"
    r"google(?:\s+\w+){1,8}|"
    r"look\s+(?:it\s+)?up\s+online|"
    r"find\s+(?:on\s+)?(?:the\s+)?(?:web|internet)|"
    r"find\s+(?:the\s+)?(?:song\s+)?lyrics|"
    r"find\s+(?:the\s+)?(?:lyrics|song\s+text)|"
    r"(?:song|lyrics)\s+text|"
    r"fetch\s+(?:the\s+)?(?:url|page|site)|"
    r"web[\s-]?search\w*"
    r")",
    re.IGNORECASE | re.UNICODE,
)

_FETCH_INTENT_RE = re.compile(
    r"(?:"
    r"go\s+to|visit(?:\s+the)?|browse\s+to|"
    r"read\s+(?:this\s+)?(?:page|url|site)|"
    r"open\s+(?:this\s+)?(?:url|page|link|site)|"
    r"quote\s+(?:the\s+)?(?:first\s+)?(?:post|page)"
    r")",
    re.IGNORECASE | re.UNICODE,
)

_URL_RE = re.compile(
    r"https?://[^\s<>\"')\]]+",
    re.IGNORECASE,
)

_RELATIVE_PATH_RE = re.compile(r"^/[\w./-]*$")

_STRIP_PREFIX_RE = re.compile(
    r"^(?:"
    r"search\s+(?:the\s+)?(?:web|internet|online)\s*(?:for\s+)?[:,\s]*|"
    r"google(?:\s+\w+){1,8}\s*[:,\s]*|"
    r"find\s+(?:on\s+)?(?:the\s+)?(?:web|internet)\s*(?:for\s+)?[:,\s]*"
    r")",
    re.IGNORECASE | re.UNICODE,
)


def extract_http_urls(text: str) -> list[str]:
    """Public http(s) URLs in user text (deduped, order preserved)."""
    seen: set[str] = set()
    out: list[str] = []
    for m in _URL_RE.finditer(text or ""):
        u = m.group(0).rstrip(".,;:!?)'\"")
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out


def prior_human_texts(
    messages: list | None, *, exclude_last: bool = False
) -> list[str]:
    """Plain-text HumanMessage contents from checkpoint messages."""
    return _human_texts_from_messages(messages, exclude_last=exclude_last)


def _human_texts_from_messages(
    messages: list | None, *, exclude_last: bool = False
) -> list[str]:
    texts: list[str] = []
    for m in messages or []:
        if not isinstance(m, HumanMessage):
            continue
        content = getattr(m, "content", "") or ""
        if isinstance(content, list):
            content = " ".join(
                b.get("text", "") if isinstance(b, dict) else str(b) for b in content
            )
        text = str(content).strip()
        if text:
            texts.append(text)
    if exclude_last and texts:
        return texts[:-1]
    return texts


def resolve_page_url(text: str, prior_human: list[str] | None = None) -> str | None:
    """First URL in text, or join a path like ``/b`` with a prior message URL."""
    urls = extract_http_urls(text)
    if urls:
        return urls[0]
    path = (text or "").strip()
    if path.startswith("/") and prior_human:
        path_only = path.strip("/")
        for prior in reversed(prior_human):
            for base in extract_http_urls(prior):
                base_stripped = base.rstrip("/")
                base_path = urlparse(base_stripped).path.rstrip("/")
                if path_only and base_path.endswith(path_only):
                    return base_stripped
                return urljoin(base, path.lstrip("/"))
    if _FETCH_INTENT_RE.search(text or "") and prior_human:
        for prior in reversed(prior_human):
            prior_urls = extract_http_urls(prior)
            if prior_urls:
                return prior_urls[0]
    return None


def wants_url_fetch(text: str, prior_human: list[str] | None = None) -> bool:
    """Prefer ``fetch_url`` over ``search_web`` for this turn."""
    if resolve_page_url(text, prior_human):
        return True
    if _FETCH_INTENT_RE.search(text or "") and prior_human:
        return any(extract_http_urls(p) for p in reversed(prior_human))
    return False


def is_web_search_intent(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    if _WEB_INTENT_RE.search(t):
        return True
    if extract_http_urls(t):
        return True
    return False


def is_web_search_intent_with_context(
    text: str, messages: list[BaseMessage] | list | None = None
) -> bool:
    """Web intent including relative paths that need a prior URL in the thread."""
    if is_web_search_intent(text):
        return True
    path = (text or "").strip()
    if _RELATIVE_PATH_RE.match(path):
        prior = _human_texts_from_messages(messages, exclude_last=True)
        return resolve_page_url(path, prior) is not None
    if _FETCH_INTENT_RE.search(text or ""):
        prior = _human_texts_from_messages(messages, exclude_last=True)
        return bool(prior) and wants_url_fetch(text, prior)
    return False


def _search_query_cap() -> int:
    from src.config import settings

    try:
        cap = int(settings.web_search_query_max_chars)
    except (TypeError, ValueError):
        cap = 160
    return max(40, min(cap, 500))


def _trim_search_query(q: str) -> str:
    text = re.sub(r"\s+", " ", (q or "").strip())
    if not text:
        return ""
    cap = _search_query_cap()
    if len(text) <= cap:
        return text
    cut = text[:cap]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut.strip() or text[:cap].strip()


_FX_QUERY_RE = re.compile(
    r"(?:" r"exchange\s+rate|conversion|convert|" r"usd|eur|gbp|thb|rub|uah|btc" r")",
    re.IGNORECASE | re.UNICODE,
)


def enhance_web_search_query(query: str, user_message: str = "") -> str:
    """Bias FX / rate questions toward verifiable search phrases."""
    q = (query or "").strip()
    src = f"{user_message} {q}"
    if not q or not _FX_QUERY_RE.search(src):
        return q
    low = q.casefold()
    if "today" not in low:
        q = f"{q} today"
    if "exchange rate" not in low:
        q = f"{q} exchange rate"
    return _trim_search_query(q)


def _focus_search_query(q: str) -> str:
    """Drop trailing USER commentary; keep the search topic."""
    text = re.sub(r"\s+", " ", (q or "").strip())
    if not text:
        return ""
    # First sentence/clause only — drop trailing commentary after the topic.
    text = re.split(r"[.!?;]\s+", text, maxsplit=1)[0].strip()
    text = re.sub(
        r"^(?:what\s+is)\s+",
        "",
        text,
        flags=re.IGNORECASE | re.UNICODE,
    )
    return text.strip(" :—-–.,!?")


def suggest_web_search_query(text: str) -> str:
    """Focused query for search engines (strip address + intent phrasing)."""
    raw = (text or "").strip()
    if not raw:
        return ""

    q = raw
    m = _WEB_INTENT_RE.search(raw)
    if m:
        q = raw[m.end() :].strip() or raw[m.start() : m.end()].strip()
    else:
        q = _STRIP_PREFIX_RE.sub("", raw)

    q = re.sub(
        r"^(?:urgent|please|now|and\s+now)[,\s]+",
        "",
        q,
        flags=re.IGNORECASE,
    )

    from src.speaker import address_prefix_pattern

    q = address_prefix_pattern().sub("", q.strip())
    q = q.strip(" :—-–.,!?")
    for u in extract_http_urls(q):
        q = q.replace(u, " ").strip()
    q = re.sub(r"\s+", " ", q).strip()
    q = _focus_search_query(q)
    q = _trim_search_query(q)
    q = q or _trim_search_query(_focus_search_query(raw)) or raw
    return enhance_web_search_query(q, raw)
