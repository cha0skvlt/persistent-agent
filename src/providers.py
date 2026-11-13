"""Resolve the chat and embedding providers from settings.

Two independent channels:

- ``LLM_PROVIDER``   — chat: ``ollama`` or ``openai`` (vLLM, OpenAI, any compatible server)
- ``EMBED_PROVIDER`` — embeddings: ``ollama`` or ``openai``

They can differ: a cloud chat model with local embeddings, or the reverse.
Everything that needs a base URL, a key, or a model name asks this module
instead of reading the environment directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.config import Settings

ProviderKind = Literal["ollama", "openai"]


@dataclass(frozen=True)
class ProviderConfig:
    """One resolved endpoint: kind, base URL, optional key, model name."""

    kind: ProviderKind
    base_url: str
    api_key: str | None
    model: str

    @property
    def is_openai(self) -> bool:
        return self.kind == "openai"

    @property
    def openai_base_url(self) -> str:
        """Base URL with the ``/v1`` suffix the OpenAI SDK expects."""
        base = self.base_url.rstrip("/")
        if base.endswith("/v1"):
            return base
        return f"{base}/v1"

    @property
    def chat_completions_url(self) -> str:
        return f"{self.openai_base_url}/chat/completions"

    @property
    def embeddings_url(self) -> str:
        return f"{self.openai_base_url}/embeddings"

    @property
    def models_url(self) -> str:
        return f"{self.openai_base_url}/models"


def _clean(value: str | None) -> str:
    return (value or "").strip()


def chat_provider(settings: Settings) -> ProviderConfig:
    """Resolved chat endpoint. ``LLM_BASE_URL`` wins over ``OLLAMA_BASE_URL``."""
    kind: ProviderKind = "openai" if settings.llm_provider == "openai" else "ollama"
    base = _clean(settings.llm_base_url) or _clean(settings.ollama_base_url)
    key = _clean(settings.llm_api_key) or None
    return ProviderConfig(
        kind=kind,
        base_url=base,
        api_key=key if kind == "openai" else None,
        model=_clean(settings.reasoning_model),
    )


def embed_provider(settings: Settings) -> ProviderConfig:
    """Resolved embedding endpoint. Falls back to the chat URL, then Ollama."""
    kind: ProviderKind = "openai" if settings.embed_provider == "openai" else "ollama"
    base = (
        _clean(settings.embed_base_url)
        or _clean(settings.llm_base_url)
        or _clean(settings.ollama_base_url)
    )
    key = _clean(settings.embed_api_key) or _clean(settings.llm_api_key) or None
    return ProviderConfig(
        kind=kind,
        base_url=base,
        api_key=key if kind == "openai" else None,
        model=_clean(settings.embed_model),
    )


def embed_dim(settings: Settings) -> int:
    """Configured embedding dimension (must match the ``vector(N)`` columns)."""
    return int(settings.embed_dim)


__all__ = [
    "ProviderConfig",
    "ProviderKind",
    "chat_provider",
    "embed_dim",
    "embed_provider",
]
