"""LangGraph tools: web search, URL fetch, local Playwright browse."""

from __future__ import annotations

from langchain_core.tools import tool

from src.tools.browse_web import browse_web_impl
from src.tools.web_core import fetch_url_text, search_web_results


@tool
def search_web(query: str) -> str:
    """Search the public web via local SearXNG then keyless fallbacks (no paid APIs, no Docker)."""
    return search_web_results(query)


@tool
def fetch_url(url: str) -> str:
    """Fetch one page WHOLE: full visible text of a public http(s) URL.

    Handles JS pages too — static HTML first, automatic local browser render
    when the page is a JS shell or blocks plain HTTP. One call = whole page.
    """
    return fetch_url_text(url)


@tool
def browse_web(action: str, target: str = "") -> str:
    """Local Playwright browser (no cloud). Returns full visible page text.

    Actions:
    - open: target=absolute http(s) URL
    - follow: target=URL or relative path
    - click: target=CSS selector or visible link text
    - extract: re-read current page text
    - close: end session
    Use ONLY when you must interact (follow links, click). For reading a page —
    even a JS-heavy one — fetch_url alone is enough.
    """
    return browse_web_impl(action=action, target=target)
