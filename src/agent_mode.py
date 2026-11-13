"""Agent runtime mode — operator sleep/wake switch.

File-backed so the API process (Telegram, reverie loop) and the Heart process
(outreach delivery) share one durable flag without a DB migration. Corrupt or
missing state file fails open to ``awake`` — a broken flag must never brick
the agent.

Sleep also unloads Ollama models (``keep_alive=0`` — same effect as
``make sleep`` / ``ollama stop``) so wired RAM is actually freed; wake
pre-warms the chat model so the first reply is not cold-load slow.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path

import httpx

from src.config import settings
from src.obs.logs import log_event

_log = logging.getLogger(__name__)

MODE_AWAKE = "awake"
MODE_SLEEP = "sleep"
_VALID_MODES = frozenset({MODE_AWAKE, MODE_SLEEP})


def _mode_path() -> Path:
    raw = (getattr(settings, "agent_mode_path", None) or "").strip()
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ".cache" / "persistent-agent" / "agent_mode.json"


def get_mode() -> str:
    try:
        data = json.loads(_mode_path().read_text(encoding="utf-8"))
        mode = str(data.get("mode") or "").strip().lower()
        if mode in _VALID_MODES:
            return mode
    except FileNotFoundError:
        pass
    except Exception as exc:
        _log.warning("agent mode read failed (%s); defaulting to awake", exc)
    return MODE_AWAKE


def is_asleep() -> bool:
    return get_mode() == MODE_SLEEP


def set_mode(mode: str, *, source: str = "operator") -> str:
    """Persist mode atomically (tmp + rename). Returns the applied mode."""
    m = (mode or "").strip().lower()
    if m not in _VALID_MODES:
        raise ValueError(f"invalid agent mode: {mode!r}")
    path = _mode_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"mode": m}), encoding="utf-8")
    os.replace(tmp, path)
    log_event("agent.mode", mode=m, source=source)
    return m


def _ollama_base() -> str:
    return str(settings.ollama_base_url or "").rstrip("/")


def loaded_ollama_models() -> list[str]:
    """Names of models currently resident in Ollama (``/api/ps``)."""
    resp = httpx.get(f"{_ollama_base()}/api/ps", timeout=10.0)
    resp.raise_for_status()
    data = resp.json()
    out: list[str] = []
    for m in data.get("models") or []:
        name = str((m or {}).get("name") or "").strip()
        if name:
            out.append(name)
    return out


def unload_ollama_models() -> list[str]:
    """Unload every resident Ollama model (frees wired RAM). Fail-soft.

    ``keep_alive=0`` on an empty generate is the documented unload path —
    the same thing ``ollama stop`` does.
    """
    try:
        models = loaded_ollama_models()
    except Exception as exc:
        _log.warning("ollama ps failed — skip model unload: %s", exc)
        return []
    unloaded: list[str] = []
    for name in models:
        try:
            httpx.post(
                f"{_ollama_base()}/api/generate",
                json={"model": name, "keep_alive": 0},
                timeout=60.0,
            ).raise_for_status()
            unloaded.append(name)
        except Exception as exc:
            _log.warning("ollama unload via API failed model=%s: %s", name, exc)
            # Same command scripts/sleep.sh uses — proven console path.
            try:
                subprocess.run(
                    ["ollama", "stop", name],
                    check=True,
                    capture_output=True,
                    timeout=60.0,
                )
                unloaded.append(name)
            except Exception as exc2:
                _log.warning("ollama stop CLI failed model=%s: %s", name, exc2)
    log_event("agent.mode.models_unloaded", models=unloaded, n=len(unloaded))
    return unloaded


def prewarm_chat_model() -> bool:
    """Load the chat model back into RAM (wake path). Fail-soft, blocking."""
    model = (settings.reasoning_model or "").strip()
    if not model:
        return False
    try:
        httpx.post(
            f"{_ollama_base()}/api/generate",
            json={"model": model, "keep_alive": settings.llm_keep_alive},
            timeout=600.0,
        ).raise_for_status()
        log_event("agent.mode.model_prewarmed", model=model)
    except Exception as exc:
        _log.warning("ollama prewarm failed model=%s: %s", model, exc)
        return False
    # /make sleep issued while the model was still loading — honor it.
    if is_asleep():
        unload_ollama_models()
        return False
    return True


__all__ = [
    "MODE_AWAKE",
    "MODE_SLEEP",
    "get_mode",
    "is_asleep",
    "loaded_ollama_models",
    "prewarm_chat_model",
    "set_mode",
    "unload_ollama_models",
]
