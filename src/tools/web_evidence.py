"""Format live web prefetch/tool output for the agent system context."""

from __future__ import annotations

import re

from src.config import settings
from src.graph.context_budget import truncate_tail

_WEB_EVIDENCE_HEADER = "LIVE WEB EVIDENCE"

_NO_RESULTS_RE = re.compile(
    r"^(?:Web search failed|No web results|Search query too short)",
    re.IGNORECASE,
)
_NUMBERED_HIT_RE = re.compile(r"(?m)^\s*1\.\s+\S")


def web_search_body_usable(body: str) -> bool:
    """True when prefetch/search output has at least one numbered snippet."""
    text = (body or "").strip()
    if not text or _NO_RESULTS_RE.match(text):
        return False
    if not text.lower().startswith("web search:"):
        return False
    return bool(_NUMBERED_HIT_RE.search(text))


def web_fetch_body_usable(body: str) -> bool:
    """True when fetch returned non-trivial page text."""
    text = (body or "").strip()
    if not text:
        return False
    low = text.lower()
    if low.startswith("fetch failed") or low.startswith("url rejected"):
        return False
    if "disabled" in low and "web_fetch" in low:
        return False
    if "unsupported content-type" in low or "no readable text" in low:
        return False
    if "too large" in low:
        return False
    if re.search(r"^HTTP \d{3}\b", text):
        return False
    return len(text) >= 80


def extract_web_body(mem_line: str) -> tuple[str, str]:
    """Return (kind, body) for a ``[WEB] …`` memory line."""
    text = (mem_line or "").strip()
    if text.startswith("[WEB] fetch_url:"):
        parts = text.split("\n", 1)
        return "fetch_url", (parts[1].strip() if len(parts) > 1 else "")
    if text.startswith("[WEB] search_web:"):
        parts = text.split("\n", 1)
        return "search_web", (parts[1].strip() if len(parts) > 1 else "")
    if text.startswith("[WEB]"):
        return "web", text.removeprefix("[WEB]").strip()
    return "", text


def web_memory_usable(mem_line: str) -> bool:
    kind, body = extract_web_body(mem_line)
    if kind == "search_web":
        return web_search_body_usable(body)
    if kind == "fetch_url":
        return web_fetch_body_usable(body)
    return False


def _clip_evidence_body(body: str) -> str:
    limit = int(settings.web_evidence_max_chars)
    text = (body or "").strip()
    if limit <= 0 or len(text) <= limit:
        return text
    return truncate_tail(text, limit)


def format_web_sys_blocks(memories: list | None) -> list[str]:
    """System-message sections for ``[WEB]`` retrieval lines (one header, no dup)."""
    entries: list[tuple[str, str, str]] = []
    for mem in memories or []:
        if not isinstance(mem, str) or not mem.strip().startswith("[WEB]"):
            continue
        kind, body = extract_web_body(mem)
        if not body:
            continue
        body = _clip_evidence_body(body)
        label = (
            "search_web"
            if kind == "search_web"
            else "fetch_url" if kind == "fetch_url" else "web"
        )
        status = "usable" if web_memory_usable(mem) else "weak"
        entries.append((label, status, body))
    usable_only = [e for e in entries if e[1] == "usable"]
    if not usable_only:
        return []
    sections = [
        f"[{label}, prefetch {status}]\n{body}" for label, status, body in usable_only
    ]
    return [f"{_WEB_EVIDENCE_HEADER.strip()}\n\n" + "\n\n".join(sections)]
