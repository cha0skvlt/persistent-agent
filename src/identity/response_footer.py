"""Optional signed footer on agent replies (``SIGN_RESPONSES``)."""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage

from src.config import settings
from src.speaker import speaker_label

_log = logging.getLogger(__name__)


def maybe_append_response_signature(message: AIMessage) -> AIMessage:
    """Append ``— Agent / sig:…`` when ``SIGN_RESPONSES`` and identity is available."""
    if not settings.sign_responses:
        return message
    if getattr(message, "tool_calls", None):
        return message
    content = getattr(message, "content", "") or ""
    if not isinstance(content, str) or not content.strip():
        return message
    try:
        from src.identity import get_signer

        sig = get_signer().sign(content)
        if not sig:
            _log.warning("identity.footer: sign returned empty — reply unsigned")
            return message
        new_content = content.rstrip() + f"\n\n— {speaker_label()} / sig:{sig}"
        return AIMessage(content=new_content, tool_calls=message.tool_calls)
    except Exception as exc:
        _log.warning("identity.footer: signer failed — reply unsigned: %s", exc)
        return message
