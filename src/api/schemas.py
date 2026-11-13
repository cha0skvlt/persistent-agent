"""Pydantic request/response models for the HTTP API."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field, field_validator

_THREAD_ID_MAX = 128
# NUL plus every C0/C1 control character. A thread id becomes a checkpoint key
# and a log field; control bytes have no business in either.
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=16_000)
    thread_id: str | None = Field(
        default=None,
        max_length=_THREAD_ID_MAX,
        description="LangGraph checkpoint thread (default: api default thread).",
    )

    @field_validator("thread_id")
    @classmethod
    def _validate_thread_id(cls, v: str | None) -> str | None:
        if v is None:
            return None
        s = v.strip()
        if not s:
            raise ValueError("thread_id must not be blank")
        if _CONTROL_RE.search(s):
            raise ValueError("thread_id must not contain control characters")
        return s


class ChatResponse(BaseModel):
    reply: str
    thread_id: str


class HealthResponse(BaseModel):
    status: str
    agent_id: str
    telegram_polling: bool
