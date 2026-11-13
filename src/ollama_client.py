"""Single Ollama HTTP client derived from settings (host for embeddings + health)."""

from __future__ import annotations

import ollama

from src.config import settings


def ollama_http_client() -> ollama.Client:
    """Return a Client pointing at ``settings.ollama_base_url`` (no global env reliance)."""
    base = settings.ollama_base_url.strip().rstrip("/")
    return ollama.Client(host=base)
