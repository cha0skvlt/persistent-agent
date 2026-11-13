"""Checkpoint serde: explicit LangChain Reviver allowlist (no LC deprecation noise)."""

from __future__ import annotations

import subprocess
import sys

import langgraph.checkpoint.serde.jsonplus as jsonplus
import pytest
from langchain_core.load.load import Reviver

from src.graph.checkpoint_serde import get_checkpoint_serde


@pytest.mark.unit
def test_checkpoint_serde_uses_core_reviver():
    serde = get_checkpoint_serde()
    assert serde is get_checkpoint_serde()
    expected = Reviver(allowed_objects="core").allowed_class_paths
    assert jsonplus.LC_REVIVER.allowed_class_paths == expected


@pytest.mark.unit
def test_ensure_reviver_core_default_is_idempotent():
    from src.graph import checkpoint_serde

    checkpoint_serde._ensure_reviver_core_default()
    checkpoint_serde._ensure_reviver_core_default()


@pytest.mark.unit
def test_reviver_default_init_uses_checkpoint_allowlist():
    from src.graph import checkpoint_serde

    checkpoint_serde._ensure_reviver_core_default()
    reviver = Reviver()
    assert reviver.allowed_class_paths


@pytest.mark.unit
def test_checkpoint_serde_import_emits_no_allowed_objects_warning():
    code = (
        "import warnings; "
        "from langchain_core._api.deprecation import LangChainPendingDeprecationWarning; "
        "warnings.simplefilter('error', LangChainPendingDeprecationWarning); "
        "import src.graph.checkpoint_serde"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(__import__("tests.support.paths", fromlist=["REPO_ROOT"]).REPO_ROOT),
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
