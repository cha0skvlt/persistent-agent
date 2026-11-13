"""Embedding transport: Ollama or any OpenAI-compatible server.

The provider is resolved by :mod:`src.providers`; this module only knows how to
call the two wire formats. Retries, input truncation, and context-overflow
shrink live here so every caller gets the same behaviour.
"""

from __future__ import annotations

import logging
import time

import httpx

from src.config import Settings, get_settings
from src.providers import ProviderConfig, embed_provider

_log = logging.getLogger(__name__)

_MAX_INPUT_CHARS = 2048
_MAX_RETRIES = 5
_RETRY_SLEEP_BASE = 0.2
_RETRY_SLEEP_CAP = 2.0
_INTERACTIVE_MAX_RETRIES = 2
_INTERACTIVE_RETRY_SLEEP_CAP = 0.5


class EmbeddingError(RuntimeError):
    """Embedding request failed after retries."""


def _input_exceeded(exc: Exception) -> bool:
    """True when the server rejected the prompt for length (non-transient for same text)."""
    s = str(exc).lower()
    needles = (
        "context length",
        "input length exceeds",
        "exceeds the context",
        "sequence length",
        "too many tokens",
        "token limit",
        "maximum context",
    )
    return any(n in s for n in needles)


def _ollama_embed(provider: ProviderConfig, text: str) -> list[float]:
    import ollama

    client = ollama.Client(host=provider.base_url.rstrip("/"))
    models = [provider.model]
    if ":" not in provider.model:
        models.append(f"{provider.model}:latest")
    last: Exception | None = None
    for model in models:
        try:
            res = client.embeddings(model=model, prompt=text)
            emb = (res or {}).get("embedding")
            if isinstance(emb, list) and emb:
                return [float(x) for x in emb]
            last = ValueError(f"empty embedding from {model}")
        except Exception as exc:
            last = exc
    raise last or EmbeddingError("ollama embeddings failed")


def _openai_embed(provider: ProviderConfig, text: str) -> list[float]:
    headers = {"Content-Type": "application/json"}
    if provider.api_key:
        headers["Authorization"] = f"Bearer {provider.api_key}"
    payload = {"model": provider.model, "input": text}
    with httpx.Client(timeout=60.0) as client:
        resp = client.post(provider.embeddings_url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
    items = data.get("data") or []
    if not items:
        raise ValueError("embeddings response has no data")
    emb = items[0].get("embedding")
    if not isinstance(emb, list) or not emb:
        raise ValueError("embeddings response has no vector")
    return [float(x) for x in emb]


def embed_text(
    text: str,
    *,
    interactive: bool = False,
    settings: Settings | None = None,
) -> list[float]:
    """Embed one string through the configured provider."""
    s = settings or get_settings()
    provider = embed_provider(s)
    prompt = (text or "").strip()
    if not prompt:
        raise ValueError("Empty prompt for embedding")
    if len(prompt) > _MAX_INPUT_CHARS:
        _log.debug("embed input truncated %s -> %s", len(prompt), _MAX_INPUT_CHARS)
        prompt = prompt[:_MAX_INPUT_CHARS]

    max_retries = _INTERACTIVE_MAX_RETRIES if interactive else _MAX_RETRIES
    sleep_cap = _INTERACTIVE_RETRY_SLEEP_CAP if interactive else _RETRY_SLEEP_CAP

    last: Exception | None = None
    for attempt in range(max_retries):
        try:
            if provider.is_openai:
                return _openai_embed(provider, prompt)
            return _ollama_embed(provider, prompt)
        except Exception as exc:
            last = exc
            if _input_exceeded(exc) and len(prompt) > 256:
                prompt = prompt[: max(256, len(prompt) // 2)]
                continue
            if attempt + 1 >= max_retries:
                break
            time.sleep(min(sleep_cap, _RETRY_SLEEP_BASE * (2**attempt)))
    raise EmbeddingError(str(last) if last else "embeddings failed")


__all__ = ["EmbeddingError", "embed_text"]
