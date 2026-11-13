"""Boot dependency checks (Ollama tags, embeddings, Postgres)."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


class _Seq:
    def __len__(self):
        return 2

    def __iter__(self):
        return iter([0.3, 0.4])


@pytest.mark.parametrize(
    "payload,expected",
    [
        ({"embedding": (0.1, 0.2)}, [0.1, 0.2]),
        ({"embedding": _Seq()}, [0.3, 0.4]),
    ],
)
def test_extract_embedding_vector_accepts_sequences(payload, expected):
    import src.boot_checks as bc

    assert bc.extract_embedding_vector(payload) == expected


def test_extract_embedding_vector_rejects_empty():
    import src.boot_checks as bc

    with pytest.raises(ValueError, match="no vector"):
        bc.extract_embedding_vector({"embedding": []})


@pytest.mark.parametrize(
    "side_effect,match",
    [
        (RuntimeError("bad embed"), "Embeddings unavailable"),
        (None, "embed dim=0"),
    ],
)
def test_assert_embeddings_ready_failures(monkeypatch, side_effect, match):
    import src.boot_checks as bc
    import src.embeddings as emb_mod

    def _fake_embed(*a, **k):
        if side_effect is None:
            return []
        raise side_effect

    monkeypatch.setattr(emb_mod, "embed_text", _fake_embed)
    with pytest.raises(RuntimeError, match=match):
        bc.assert_embeddings_ready()


def test_assert_embeddings_ready_wrong_dim(monkeypatch):
    import src.boot_checks as bc
    import src.embeddings as emb_mod

    monkeypatch.setattr(emb_mod, "embed_text", lambda *a, **k: [0.0] * 64)
    with pytest.raises(RuntimeError, match="expected 768"):
        bc.assert_embeddings_ready()


def test_log_llm_ollama_budget(caplog):
    import src.boot_checks as bc

    caplog.set_level("INFO", logger=bc.__name__)
    bc.log_llm_ollama_budget()
    assert any(
        "num_ctx=" in r.message and "num_predict=" in r.message for r in caplog.records
    )


def test_verify_boot_warns_low_num_predict(monkeypatch, caplog):
    import src.boot_checks as bc
    from src.config import Settings

    monkeypatch.setattr(bc, "assert_llm_ready", lambda: None)
    monkeypatch.setattr(bc, "assert_embeddings_ready", lambda: None)
    monkeypatch.setattr(bc, "assert_database_ready", lambda: None)
    monkeypatch.setattr(
        "src.api.telegram_bot.assert_telegram_boot_allowlist", lambda: None
    )
    low = Settings(_env_file=None, llm_num_predict=1024)
    monkeypatch.setattr(bc, "settings", low)
    caplog.set_level("WARNING", logger="src.config")
    caplog.set_level("INFO", logger=bc.__name__)
    bc.verify_boot_dependencies()
    assert any("num_predict too low" in r.message for r in caplog.records)
    assert any("Ollama chat options:" in r.message for r in caplog.records)


def test_verify_boot_logs_sampling_budget(monkeypatch, caplog):
    import src.boot_checks as bc

    monkeypatch.setattr(bc, "assert_llm_ready", lambda: None)
    monkeypatch.setattr(bc, "assert_embeddings_ready", lambda: None)
    monkeypatch.setattr(bc, "assert_database_ready", lambda: None)
    monkeypatch.setattr(
        "src.api.telegram_bot.assert_telegram_boot_allowlist", lambda: None
    )
    caplog.set_level("INFO", logger="src.boot_checks")
    bc.verify_boot_dependencies()
    assert any("Ollama chat options:" in r.message for r in caplog.records)
