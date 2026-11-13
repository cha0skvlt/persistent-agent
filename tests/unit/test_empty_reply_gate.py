"""Empty-reply gate: promote spoken prose from reasoning into content."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from src.graph.empty_reply_gate import materialize_spoken_reply

_PROMOTED = (
    "The web is silent on that manual — send a link or another query, "
    "otherwise there is nothing to quote and I will not make it up."
)


def test_materialize_promotes_reasoning_when_content_empty():
    msg = AIMessage(
        content="",
        additional_kwargs={
            "reasoning_content": f"Final Polish:\n{_PROMOTED}\nCheck Constraints:\n* No markdown"
        },
    )
    out = materialize_spoken_reply(msg)
    assert "The web is silent" in out.content
    assert out.additional_kwargs.get("reasoning_content")


def test_materialize_promotes_when_content_is_empty_claim():
    msg = AIMessage(
        content="nothing",
        additional_kwargs={
            "reasoning_content": (
                f"Final Polish:\n{_PROMOTED}\n" "Check Constraints:\n* No markdown"
            )
        },
    )
    out = materialize_spoken_reply(msg)
    assert out.content.lower() != "nothing"
    assert "web" in out.content.lower() or "manual" in out.content.lower()


def test_materialize_keeps_distinct_spoken_content():
    msg = AIMessage(
        content="Already said it out loud.",
        additional_kwargs={"reasoning_content": "thinking about something else"},
    )
    out = materialize_spoken_reply(msg)
    assert out.content == "Already said it out loud."


def test_materialize_noop_when_nothing_to_promote():
    msg = AIMessage(content="nothing")
    out = materialize_spoken_reply(msg)
    assert out.content == "nothing"
