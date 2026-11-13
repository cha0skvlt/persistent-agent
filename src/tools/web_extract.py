"""HTML → full visible page text (site-agnostic)."""

from __future__ import annotations

import re
from html import unescape
from urllib.parse import urlparse

from lxml import html as lxml_html

from src.config import settings

_SCRIPT_DROP = frozenset({"script", "style", "noscript", "svg", "iframe", "template"})
_WS_RE = re.compile(r"[ \t\f\v]+")
_BLANK_RE = re.compile(r"\n{3,}")
_BLOCK_TAGS = frozenset(
    {
        "p",
        "div",
        "br",
        "li",
        "tr",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "blockquote",
        "section",
        "article",
        "hr",
    }
)


def _clip(text: str, max_chars: int) -> str:
    if max_chars > 0 and len(text) > max_chars:
        return text[:max_chars] + "…"
    return text


def compact_visible_text(text: str) -> str:
    """Drop empty-line bombs from SPA/chrome markup; keep at most one blank between blocks."""
    lines: list[str] = []
    blank_run = 0
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            blank_run += 1
            if blank_run <= 1:
                lines.append("")
            continue
        blank_run = 0
        lines.append(line)
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def _drop_script_nodes(tree: lxml_html.HtmlElement) -> None:
    for el in list(tree.iter()):
        tag = getattr(el, "tag", "")
        if isinstance(tag, str) and tag.lower() in _SCRIPT_DROP:
            parent = el.getparent()
            if parent is not None:
                parent.remove(el)


def extract_page_text(html: str, *, max_chars: int | None = None) -> str:
    """Full visible page text from HTML body (any site)."""
    limit = max_chars if max_chars is not None else int(settings.web_fetch_max_chars)
    raw = html or ""
    if not raw.strip():
        return ""
    try:
        tree = lxml_html.fromstring(raw)
    except Exception:
        text = unescape(re.sub(r"<[^>]+>", "\n", raw))
        text = _BLANK_RE.sub("\n\n", _WS_RE.sub(" ", text)).strip()
        return _clip(text, limit)
    _drop_script_nodes(tree)
    bodies = tree.xpath("//body")
    root = bodies[0] if bodies else tree
    parts: list[str] = []
    for el in root.iter():
        tag = getattr(el, "tag", "")
        if not isinstance(tag, str):
            continue
        if el.text:
            parts.append(el.text)
        if el.tail:
            parts.append(el.tail)
        if tag.lower() in _BLOCK_TAGS:
            parts.append("\n")
    text = "".join(parts)
    text = _WS_RE.sub(" ", text)
    text = _BLANK_RE.sub("\n\n", text)
    text = compact_visible_text(text)
    return _clip(text, limit)


def extract_readable_text(html: str, *, max_chars: int | None = None) -> str:
    """Alias used by ``html_to_text`` / fetch path."""
    return extract_page_text(html, max_chars=max_chars)


def redirect_note(requested: str, final: str) -> str:
    """One-line note when the landing URL path differs from the request."""
    req = (requested or "").strip()
    fin = (final or "").strip()
    if not req or not fin:
        return ""
    try:
        rp = urlparse(req)
        fp = urlparse(fin)
    except Exception:
        return ""
    req_path = (rp.path or "/").rstrip("/") or "/"
    fin_path = (fp.path or "/").rstrip("/") or "/"
    if req.rstrip("/") == fin.rstrip("/"):
        return ""
    if req_path == fin_path and (rp.netloc or "").lower() == (fp.netloc or "").lower():
        return ""
    return f"(redirected from {req})"


__all__ = [
    "compact_visible_text",
    "extract_page_text",
    "extract_readable_text",
    "redirect_note",
]
