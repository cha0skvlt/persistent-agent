"""Provider resolution, embedding transport, chat factory, and boot probes."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.config import Settings
from src.providers import chat_provider, embed_dim, embed_provider

# --- resolution -------------------------------------------------------------


@pytest.mark.unit
def test_defaults_are_local_ollama():
    s = Settings()
    chat = chat_provider(s)
    emb = embed_provider(s)
    assert chat.kind == "ollama"
    assert chat.base_url == "http://127.0.0.1:11434"
    assert chat.api_key is None
    assert emb.kind == "ollama"
    assert embed_dim(s) == 768


@pytest.mark.unit
def test_openai_chat_provider():
    s = Settings(
        llm_provider="openai",
        llm_base_url="http://127.0.0.1:8000/v1",
        llm_api_key="secret",
        reasoning_model="Qwen/Qwen2.5-7B-Instruct",
    )
    chat = chat_provider(s)
    assert chat.kind == "openai"
    assert chat.api_key == "secret"
    assert chat.model == "Qwen/Qwen2.5-7B-Instruct"
    assert chat.openai_base_url == "http://127.0.0.1:8000/v1"
    assert chat.chat_completions_url.endswith("/v1/chat/completions")


@pytest.mark.unit
def test_openai_base_url_gets_v1_suffix():
    s = Settings(llm_provider="openai", llm_base_url="http://host:8000")
    assert chat_provider(s).openai_base_url == "http://host:8000/v1"


@pytest.mark.unit
def test_openai_requires_explicit_base_url():
    with pytest.raises(ValidationError, match="LLM_BASE_URL"):
        Settings(llm_provider="openai")


@pytest.mark.unit
def test_embed_provider_falls_back_to_chat_url():
    s = Settings(
        llm_provider="openai",
        llm_base_url="http://host:8000/v1",
        llm_api_key="k",
        embed_provider="openai",
    )
    emb = embed_provider(s)
    assert emb.kind == "openai"
    assert emb.base_url == "http://host:8000/v1"
    assert emb.api_key == "k"
    assert emb.embeddings_url.endswith("/v1/embeddings")


@pytest.mark.unit
def test_embed_provider_can_differ_from_chat():
    s = Settings(
        llm_provider="openai",
        llm_base_url="http://host:8000/v1",
        embed_provider="ollama",
        ollama_base_url="http://127.0.0.1:11434",
    )
    assert chat_provider(s).kind == "openai"
    assert embed_provider(s).kind == "ollama"


@pytest.mark.unit
def test_unknown_provider_falls_back_to_ollama():
    s = Settings(llm_provider="bogus")
    assert chat_provider(s).kind == "ollama"


# --- embedding transport ----------------------------------------------------


@pytest.mark.unit
def test_embed_text_uses_openai_wire_format(monkeypatch):
    import src.embeddings as emb_mod

    seen: dict = {}

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [{"embedding": [0.1, 0.2, 0.3]}]}

    class _Client:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, json=None, headers=None):
            seen["url"] = url
            seen["json"] = json
            seen["headers"] = headers
            return _Resp()

    monkeypatch.setattr(emb_mod.httpx, "Client", _Client)
    s = Settings(
        embed_provider="openai",
        embed_base_url="http://host:8000/v1",
        embed_api_key="k",
        embed_model="text-embedding-3-small",
    )
    vec = emb_mod.embed_text("hello", settings=s)
    assert vec == [0.1, 0.2, 0.3]
    assert seen["url"] == "http://host:8000/v1/embeddings"
    assert seen["json"]["model"] == "text-embedding-3-small"
    assert seen["headers"]["Authorization"] == "Bearer k"


# --- chat factory -----------------------------------------------------------


@pytest.mark.unit
def test_make_llm_returns_ollama_by_default():
    from src.llm_factory import _AgentChatOllama, make_llm

    assert isinstance(make_llm(Settings()), _AgentChatOllama)


@pytest.mark.unit
def test_make_llm_returns_openai_client():
    from langchain_openai import ChatOpenAI

    from src.llm_factory import make_llm

    s = Settings(
        llm_provider="openai",
        llm_base_url="http://host:8000/v1",
        llm_api_key="k",
        llm_extra_body='{"top_k": 40}',
    )
    llm = make_llm(s)
    assert isinstance(llm, ChatOpenAI)


# --- boot probes ------------------------------------------------------------


@pytest.mark.unit
def test_assert_llm_ready_rejects_401(monkeypatch):
    import src.boot_checks as bc

    class _Resp:
        status_code = 401

        def raise_for_status(self):
            return None

    class _Client:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url, headers=None):
            return _Resp()

    monkeypatch.setattr(bc.httpx, "Client", _Client)
    monkeypatch.setattr(
        bc,
        "settings",
        Settings(
            llm_provider="openai",
            llm_base_url="http://host:8000/v1",
            llm_api_key="bad",
        ),
    )
    with pytest.raises(RuntimeError, match="401"):
        bc.assert_llm_ready()


@pytest.mark.unit
def test_installer_does_not_install_a_model_runtime():
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    script = (root / "install.sh").read_text(encoding="utf-8")
    assert "Does NOT install a model runtime" in script
    assert "ollama pull" not in script
    assert "brew install ollama" not in script
    assert "_pkg_install ollama" not in script


@pytest.mark.unit
def test_assert_embeddings_ready_rejects_wrong_dimension(monkeypatch):
    import src.boot_checks as bc
    import src.embeddings as emb_mod

    monkeypatch.setattr(emb_mod, "embed_text", lambda *a, **k: [0.1, 0.2, 0.3])
    monkeypatch.setattr(bc, "settings", Settings(embed_dim=768))
    with pytest.raises(RuntimeError, match="embed dim=3"):
        bc.assert_embeddings_ready()
