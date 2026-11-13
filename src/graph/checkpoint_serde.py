"""LangGraph checkpoint serde with explicit LangChain Reviver allowlist.

LangGraph's ``jsonplus`` module instantiates ``Reviver()`` at import time.
LangChain >= 1.3.3 deprecates the implicit default; we pin ``allowed_objects='core'``
(the current default, includes message types in agent checkpoints) before serde loads.
"""

from __future__ import annotations

# isort: skip_file

from langchain_core.load.load import Reviver

_CHECKPOINT_ALLOWED_OBJECTS = "core"


def _ensure_reviver_core_default() -> None:
    if getattr(Reviver, "_agent_core_default_init", False):
        return
    orig_init = Reviver.__init__

    def _reviver_init_explicit_core(self, allowed_objects=None, *args, **kwargs):
        if allowed_objects is None:
            allowed_objects = _CHECKPOINT_ALLOWED_OBJECTS
        return orig_init(self, allowed_objects, *args, **kwargs)

    Reviver.__init__ = _reviver_init_explicit_core  # type: ignore[method-assign]
    Reviver._agent_core_default_init = True  # type: ignore[attr-defined]


_ensure_reviver_core_default()

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer  # noqa: E402
import langgraph.checkpoint.serde.jsonplus as _jsonplus  # noqa: E402

_jsonplus.LC_REVIVER = Reviver(allowed_objects=_CHECKPOINT_ALLOWED_OBJECTS)

_CHECKPOINT_SERDE = JsonPlusSerializer()


def get_checkpoint_serde() -> JsonPlusSerializer:
    """Shared serializer for ``MemorySaver`` / ``PostgresSaver``."""
    return _CHECKPOINT_SERDE
