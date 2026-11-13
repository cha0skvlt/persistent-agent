"""AppRuntime process resets for tests."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def test_reset_process_state_for_tests_smoke():
    from src.app_runtime import reset_process_state_for_tests
    from src.embed_status import embed_unavailable_detail, mark_embed_unavailable
    from src.llm_factory import clear_llm_cache

    mark_embed_unavailable("probe")
    assert embed_unavailable_detail() == "probe"
    clear_llm_cache()
    reset_process_state_for_tests(close_db=True)
    assert embed_unavailable_detail() is None


def test_clear_llm_caches_callable():
    from src.app_runtime import clear_llm_caches

    clear_llm_caches()


def test_conftest_autouse_uses_single_process_wipe():
    """P2: pytest must not reassemble resets — only ``reset_process_state_for_tests``."""
    text = Path("tests/conftest.py").read_text(encoding="utf-8")
    assert "from src.app_runtime import reset_process_state_for_tests" in text
    assert "def _reset_app_runtime():" in text
    assert "reset_graph_app" not in text
    assert "clear_llm_cache" not in text
    assert "reset_identity_signer" not in text
    assert text.count("reset_process_state_for_tests") >= 3


def test_prompt_parity_uses_shared_yaml_loader():
    import src.prompt_tool_parity as ptp
    import src.system_prompt_text as spt

    assert not hasattr(ptp, "_read_prompt_doc")
    assert callable(spt.read_yaml_doc)
