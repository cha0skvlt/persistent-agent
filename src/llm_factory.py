"""Lazy singleton chat client: Ollama or any OpenAI-compatible server."""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Any, Literal

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from src.config import Settings, get_settings
from src.providers import ProviderConfig, chat_provider

_log = logging.getLogger(__name__)

_llm_with_tools: dict[tuple[str, ...], Any] = {}
_agent_llm_with_tools: dict[tuple[str, ...], Any] = {}

# Fixed sampling — not operator knobs. Speech keeps room to talk; tools stay tight.
_LLM_TEMPERATURE = 0.9
_LLM_TOP_P = 0.95
_LLM_TOP_K = 40
_LLM_MIN_P = 0.0
_LLM_REPEAT_PENALTY = 1.0
_LLM_PRESENCE_PENALTY = 0.0

_LLM_TOOLS_TEMPERATURE = 0.3
_LLM_TOOLS_TOP_P = 0.8
_LLM_TOOLS_MIN_P = 0.05
# A thinking model spends part of this budget before it emits a tool call.
# 2048 was enough for a small model and cut the call off on a 122B.
_LLM_TOOLS_NUM_PREDICT = 4096

LlmMode = Literal["speech", "tools"]


def ollama_chat_invocation_options(settings: Settings) -> dict[str, int | float]:
    """Ollama ``options`` payload for speech — context from Settings, sampling fixed."""
    return {
        "num_ctx": int(settings.llm_num_ctx),
        "num_predict": int(settings.llm_num_predict),
        "temperature": _LLM_TEMPERATURE,
        "top_p": _LLM_TOP_P,
        "top_k": _LLM_TOP_K,
        "min_p": _LLM_MIN_P,
        "repeat_penalty": _LLM_REPEAT_PENALTY,
        "presence_penalty": _LLM_PRESENCE_PENALTY,
    }


def ollama_tools_invocation_options(settings: Settings) -> dict[str, int | float]:
    """Ollama ``options`` for tool-probe generate — cooler, no thinking budget."""
    return {
        "num_ctx": int(settings.llm_num_ctx),
        "num_predict": _LLM_TOOLS_NUM_PREDICT,
        "temperature": _LLM_TOOLS_TEMPERATURE,
        "top_p": _LLM_TOOLS_TOP_P,
        "top_k": _LLM_TOP_K,
        "min_p": _LLM_TOOLS_MIN_P,
        "repeat_penalty": _LLM_REPEAT_PENALTY,
        "presence_penalty": _LLM_PRESENCE_PENALTY,
    }


def fixed_sampling_summary() -> dict[str, int | float]:
    """Fixed speech sampling (for boot logs)."""
    return {
        "temperature": _LLM_TEMPERATURE,
        "top_p": _LLM_TOP_P,
        "top_k": _LLM_TOP_K,
        "min_p": _LLM_MIN_P,
        "repeat_penalty": _LLM_REPEAT_PENALTY,
        "presence_penalty": _LLM_PRESENCE_PENALTY,
    }


def fixed_tools_sampling_summary() -> dict[str, int | float]:
    """Fixed tool-probe sampling (for boot logs)."""
    return {
        "temperature": _LLM_TOOLS_TEMPERATURE,
        "top_p": _LLM_TOOLS_TOP_P,
        "top_k": _LLM_TOP_K,
        "min_p": _LLM_TOOLS_MIN_P,
        "num_predict": _LLM_TOOLS_NUM_PREDICT,
        "repeat_penalty": _LLM_REPEAT_PENALTY,
        "presence_penalty": _LLM_PRESENCE_PENALTY,
    }


class _AgentChatOllama(ChatOllama):
    """Ensures configured Ollama ``options`` reach each chat request."""

    _agent_options: dict[str, int | float]

    def _chat_params(
        self,
        messages: list,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        params = super()._chat_params(messages, stop=stop, **kwargs)
        opts = dict(params.get("options") or {})
        opts.update(self._agent_options)
        params["options"] = opts
        return params


def _extra_body(settings: Settings) -> dict[str, Any] | None:
    """Parse ``LLM_EXTRA_BODY`` (JSON object) for OpenAI-compatible servers."""
    raw = (settings.llm_extra_body or "").strip()
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        _log.warning("LLM_EXTRA_BODY is not valid JSON; ignoring: %s", exc)
        return None
    return data if isinstance(data, dict) and data else None


def _make_openai_llm(
    settings: Settings,
    provider: ProviderConfig,
    *,
    mode: LlmMode,
) -> Any:
    """Chat client for vLLM / OpenAI / any OpenAI-compatible endpoint."""
    from langchain_openai import ChatOpenAI

    opts = (
        ollama_tools_invocation_options(settings)
        if mode == "tools"
        else ollama_chat_invocation_options(settings)
    )
    kwargs: dict[str, Any] = {
        "model": provider.model,
        "base_url": provider.openai_base_url,
        "api_key": provider.api_key or "not-needed",
        "temperature": float(opts["temperature"]),
        "top_p": float(opts["top_p"]),
        "max_tokens": int(opts["num_predict"]),
        "presence_penalty": float(opts["presence_penalty"]),
    }
    extra = _extra_body(settings)
    if extra:
        kwargs["extra_body"] = extra
    return ChatOpenAI(**kwargs)


def make_llm(
    settings: Settings,
    *,
    for_agent: bool = False,
    mode: LlmMode = "speech",
) -> Any:
    # for_agent is kept for call-site compatibility; runtime uses one USER/AGENT model tag.
    provider = chat_provider(settings)
    if provider.is_openai:
        return _make_openai_llm(settings, provider, mode=mode)

    if mode == "tools":
        opts = ollama_tools_invocation_options(settings)
        reasoning = False
    else:
        opts = ollama_chat_invocation_options(settings)
        reasoning = for_agent and settings.llm_reasoning
    llm = _AgentChatOllama(
        model=provider.model,
        base_url=provider.base_url,
        keep_alive=settings.llm_keep_alive,
        num_ctx=int(opts["num_ctx"]),
        num_predict=int(opts["num_predict"]),
        temperature=float(opts["temperature"]),
        top_p=float(opts["top_p"]),
        top_k=int(opts["top_k"]),
        repeat_penalty=float(opts["repeat_penalty"]),
        reasoning=reasoning,
    )
    llm._agent_options = opts
    return llm


def ollama_chat_request_options(llm: ChatOllama) -> dict[str, Any]:
    """Resolved ``options`` dict for the next Ollama chat request (tests / boot introspection)."""
    return llm._chat_params([HumanMessage(content="")]).get("options") or {}


@lru_cache(maxsize=1)
def get_llm() -> ChatOllama:
    """Main chat model (reflect, summary, role review)."""
    return make_llm(get_settings())


@lru_cache(maxsize=1)
def get_agent_llm() -> ChatOllama:
    """Agent speech model; exposes Ollama native reasoning metadata."""
    return make_llm(get_settings(), for_agent=True)


@lru_cache(maxsize=1)
def get_agent_tools_llm() -> ChatOllama:
    """Agent tool-probe model — no thinking channel, cooler sampling."""
    return make_llm(get_settings(), for_agent=True, mode="tools")


def get_llm_with_tools(tools: list[Any]) -> Any:
    settings = get_settings()
    if not bool(settings.llm_supports_tools):
        # Models without native tool calling: no bind_tools; agent uses HTC recovery.
        return get_llm()
    key = tuple(
        sorted(
            str(getattr(t, "name", "")) for t in tools if str(getattr(t, "name", ""))
        )
    )
    cached = _llm_with_tools.get(key)
    if cached is not None:
        return cached
    bound = get_llm().bind_tools(tools)
    _llm_with_tools[key] = bound
    return bound


def get_agent_llm_with_tools(tools: list[Any]) -> Any:
    settings = get_settings()
    if not bool(settings.llm_supports_tools):
        # Unbound probe still uses tools sampling so HTC JSON stays cooler.
        return get_agent_tools_llm()
    key = tuple(
        sorted(
            str(getattr(t, "name", "")) for t in tools if str(getattr(t, "name", ""))
        )
    )
    cached = _agent_llm_with_tools.get(key)
    if cached is not None:
        return cached
    bound = get_agent_tools_llm().bind_tools(tools)
    _agent_llm_with_tools[key] = bound
    return bound


def clear_llm_cache() -> None:
    """Reset cached instances (pytest / multi-app in one process).

    Safe when tests monkeypatch ``get_llm`` / ``get_agent_llm`` to plain callables.
    """
    for fn in (get_llm, get_agent_llm, get_agent_tools_llm):
        clear = getattr(fn, "cache_clear", None)
        if callable(clear):
            clear()
    _llm_with_tools.clear()
    _agent_llm_with_tools.clear()


__all__ = [
    "clear_llm_cache",
    "fixed_sampling_summary",
    "fixed_tools_sampling_summary",
    "get_agent_llm",
    "get_agent_llm_with_tools",
    "get_agent_tools_llm",
    "get_llm",
    "get_llm_with_tools",
    "make_llm",
    "ollama_chat_invocation_options",
    "ollama_chat_request_options",
    "ollama_tools_invocation_options",
]
