"""Shared startup dependency checks (REPL, API)."""

from __future__ import annotations

import logging

import httpx
import psycopg

from src.config import (
    settings,
    warn_llm_num_predict_if_needed,
)
from src.ollama_client import ollama_http_client
from src.providers import chat_provider

_log = logging.getLogger(__name__)

EMBED_DIM = 768


def extract_embedding_vector(res) -> list[float]:
    """Require a non-empty embedding vector from an Ollama embeddings response."""
    emb = (res or {}).get("embedding")
    if emb is None or not hasattr(emb, "__len__") or len(emb) == 0:
        raise ValueError("embeddings returned no vector")
    if not isinstance(emb, (list, tuple)):
        return list(emb)
    return list(emb)


def assert_llm_ready() -> None:
    """Fail boot if the chat endpoint is unavailable or rejects the API key."""
    provider = chat_provider(settings)
    if provider.is_openai:
        headers: dict[str, str] = {}
        if provider.api_key:
            headers["Authorization"] = f"Bearer {provider.api_key}"
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(provider.models_url, headers=headers)
                if resp.status_code == 401:
                    raise RuntimeError(
                        f"LLM API key rejected by {provider.base_url} (401)"
                    )
                resp.raise_for_status()
        except Exception as e:
            raise RuntimeError(
                f"Chat endpoint unavailable at {provider.base_url}: {e}. "
                "The agent does not start a model server — start Ollama or vLLM yourself."
            ) from e
        return
    try:
        ollama_http_client().list()
    except Exception as e:
        raise RuntimeError(
            f"Chat endpoint unavailable at {provider.base_url}: {e}. "
            "The agent does not start a model server — start Ollama yourself."
        ) from e


def assert_embeddings_ready(*, expected_dim: int | None = None) -> None:
    """Fail boot if the embedding endpoint is unusable or the dimension is wrong."""
    from src.embeddings import embed_text

    dim = int(expected_dim or settings.embed_dim)
    try:
        vec = embed_text("ping", interactive=True)
    except Exception as e:
        raise RuntimeError(
            "Embeddings unavailable "
            f"(provider={settings.embed_provider}, model={settings.embed_model}): {e}. "
            "Start the embedding server yourself and check EMBED_BASE_URL."
        ) from e
    if len(vec) != dim:
        raise RuntimeError(
            f"embed dim={len(vec)}, expected {dim} "
            "(EMBED_DIM does not match the vector(N) columns)"
        )


def assert_database_ready() -> None:
    """Fail boot if PostgreSQL is unavailable."""
    try:
        with psycopg.connect(settings.database_url, connect_timeout=2) as conn:
            with conn.cursor() as cur:
                cur.execute("select 1")
                cur.fetchone()
    except Exception as e:
        raise RuntimeError(
            f"database unavailable ({settings.database_name}) at "
            f"{settings.database_host}:{settings.database_port}: {e}"
        ) from e


_ROLE_REVIEW_SCHEMA_MISSING = (
    "role_review enabled but dpo_gold_pairs missing — run: make migrate"
)


def assert_role_review_schema_ready() -> None:
    """Fail boot when ROLE_REVIEW=1 but ``dpo_gold_pairs`` is absent."""
    if not bool(settings.role_review):
        return
    try:
        with psycopg.connect(settings.database_url, connect_timeout=2) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT to_regclass('public.dpo_gold_pairs')")
                row = cur.fetchone()
    except Exception as e:
        raise RuntimeError(
            f"role_review schema check failed ({settings.database_name}): {e}"
        ) from e
    if row is None or row[0] is None:
        raise RuntimeError(_ROLE_REVIEW_SCHEMA_MISSING)


def log_llm_budget() -> None:
    """Emit configured chat token budgets at boot (provider-aware)."""
    from src.llm_factory import fixed_sampling_summary, fixed_tools_sampling_summary

    provider = chat_provider(settings)
    if provider.is_openai:
        _log.info(
            "LLM provider=openai base=%s model=%s max_tokens=%s",
            provider.base_url,
            provider.model,
            settings.llm_num_predict,
        )
        return

    samp = fixed_sampling_summary()
    tools = fixed_tools_sampling_summary()
    _log.info(
        "Ollama chat options: num_ctx=%s num_predict=%s temperature=%s top_p=%s "
        "top_k=%s min_p=%s model=%s",
        settings.llm_num_ctx,
        settings.llm_num_predict,
        samp["temperature"],
        samp["top_p"],
        samp["top_k"],
        samp["min_p"],
        settings.reasoning_model,
    )
    _log.info(
        "Ollama tools options: num_predict=%s temperature=%s top_p=%s min_p=%s",
        tools["num_predict"],
        tools["temperature"],
        tools["top_p"],
        tools["min_p"],
    )


def verify_boot_dependencies() -> None:
    """Ollama tags + embeddings (validates keep_alive) + Postgres + Telegram allowlist."""
    warn_llm_num_predict_if_needed(settings)
    log_llm_budget()
    binding = "native" if bool(settings.llm_supports_tools) else "htc_only"
    _log.info("llm.tools_binding=%s", binding)
    assert_llm_ready()
    assert_embeddings_ready()
    assert_database_ready()
    assert_role_review_schema_ready()
    from src.api.telegram_bot import assert_telegram_boot_allowlist

    assert_telegram_boot_allowlist()


def verify_boot_dependencies_with_retry() -> None:
    """Retry ``verify_boot_dependencies`` up to ``BOOT_RETRY_ATTEMPTS`` times.

    After reboot, launchd may start the API before Ollama has loaded the model.
    Instead of crashing in a restart loop we wait patiently, logging each attempt.
    """
    import time

    max_attempts = max(settings.boot_retry_attempts, 1)
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            verify_boot_dependencies()
            return
        except Exception as exc:
            last_exc = exc
            _log.warning(
                "boot dependency check failed (attempt %d/%d): %s",
                attempt,
                max_attempts,
                exc,
            )
            if attempt < max_attempts:
                time.sleep(settings.boot_retry_sleep_sec)
    raise RuntimeError(
        f"boot dependencies unavailable after {max_attempts} attempts"
    ) from last_exc


# Backwards-compatible aliases. Prefer the provider-neutral names in new code.
assert_ollama_ready = assert_llm_ready
assert_ollama_embeddings_ready = assert_embeddings_ready
log_llm_ollama_budget = log_llm_budget

__all__ = [
    "EMBED_DIM",
    "assert_database_ready",
    "assert_embeddings_ready",
    "assert_llm_ready",
    "assert_ollama_embeddings_ready",
    "assert_ollama_ready",
    "assert_role_review_schema_ready",
    "extract_embedding_vector",
    "log_llm_budget",
    "log_llm_ollama_budget",
    "verify_boot_dependencies",
    "verify_boot_dependencies_with_retry",
]
