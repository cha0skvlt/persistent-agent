"""Single place to load system_prompt.yml text (avoids import cycles workflow ↔ retrieval)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

_log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]


def read_yaml_doc(path: Path) -> dict[str, Any]:
    """Load a YAML mapping; missing/unreadable files become ``{}``."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
        return doc if isinstance(doc, dict) else {}
    except FileNotFoundError:
        return {}
    except OSError as e:
        _log.warning("Could not read yaml from %s: %s", path, e)
        return {}


def _read_yaml_doc(path: Path) -> dict[str, Any]:
    """Alias for callers/tests that still use the private name."""
    return read_yaml_doc(path)


def _compose_system_prompt(doc: dict[str, Any]) -> str:
    """Language rule, operating rules, tool protocol, selftest protocol."""
    parts: list[str] = []
    # Language first — it is the cheapest constraint to satisfy and the most expensive to lose.
    lang = str(doc.get("language_rule") or "").strip()
    if lang:
        parts.append(lang)
    core = str(doc.get("system") or doc.get("prompt") or "").strip()
    if core:
        parts.append(core)
    protocol = str(doc.get("tools_protocol") or "").strip()
    if protocol:
        parts.append(protocol)
    selftest = str(doc.get("selftest_protocol") or "").strip()
    if selftest:
        parts.append(selftest)
    voice = str(doc.get("voice") or "").strip()
    if voice:
        parts.append(voice)
    return "\n\n".join(parts)


def _load_yml_system_key(candidates: list[Path], label: str) -> str:
    for path in candidates:
        doc = _read_yaml_doc(path)
        text = _compose_system_prompt(doc)
        if text:
            return text
    return ""


def _system_prompt_paths() -> list[Path]:
    return [
        _REPO_ROOT / "prompts" / "agent" / "system_prompt.yml",
        _REPO_ROOT / "system_prompt.yml",  # legacy root layout
        Path("prompts/agent/system_prompt.yml"),
        Path("system_prompt.yml"),
    ]


def load_system_prompt_text() -> str:
    text = _load_yml_system_key(_system_prompt_paths(), "system_prompt")
    if not text:
        _log.warning("prompts/agent/system_prompt.yml not found; system prompt empty.")
    return text


def load_telegram_output_protocol() -> str:
    """Telegram-only reply formatting (not injected into REPL/API)."""
    for path in _system_prompt_paths():
        doc = _read_yaml_doc(path)
        text = str(doc.get("telegram_output") or "").strip()
        if text:
            return text
    return ""


def load_solitary_prompt() -> str:
    """Solitary user-message loaded from system_prompt.yml ``solitary_prompt``."""
    for path in _system_prompt_paths():
        doc = _read_yaml_doc(path)
        text = str(doc.get("solitary_prompt") or "").strip()
        if text:
            return text
    return ""
