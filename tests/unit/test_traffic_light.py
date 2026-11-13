"""Traffic light — red gate, yellow cancellable stream, resume thread."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk


@pytest.mark.unit
def test_red_background_rejected_when_live_holds_gate():
    import src.agent_runner as ar
    from src.reverie.cancel import ReverieCancelledError
    from src.reverie.turn_kind import TurnKind, turn_kind

    ar.reset_llm_turn_semaphore()
    sem = ar._get_llm_turn_semaphore()
    assert sem.acquire(blocking=False)
    try:
        with turn_kind(TurnKind.SOLITARY):
            with pytest.raises(ReverieCancelledError):
                ar.run_agent_turn_detailed("x", thread_id="bg")
    finally:
        sem.release()
        ar.reset_llm_turn_semaphore()


@pytest.mark.unit
def test_yellow_stream_aborts_on_cancel(monkeypatch):
    from src.obs import llm_invoke as li
    from src.reverie.cancel import (
        ReverieCancelledError,
        begin_reverie_cycle,
        end_reverie_cycle,
        request_reverie_cancel,
        reset_reverie_cancel_state,
    )
    from src.reverie.turn_kind import TurnKind, turn_kind

    reset_reverie_cancel_state()

    class _StreamLLM:
        def invoke(self, *_a, **_k):
            raise AssertionError("must stream")

        def stream(self, _msgs, config=None):
            for i in range(50):
                if i == 2:
                    request_reverie_cancel()
                yield AIMessageChunk(content=f"c{i}")

    token = begin_reverie_cycle()
    try:
        with turn_kind(TurnKind.SOLITARY):
            with pytest.raises(ReverieCancelledError):
                li.invoke_llm(
                    _StreamLLM(), [{"role": "user", "content": "x"}], role="agent"
                )
    finally:
        end_reverie_cycle(token)
        reset_reverie_cancel_state()


@pytest.mark.unit
def test_yellow_stream_finalizes_to_plain_aimessage():
    from langchain_core.messages import AIMessage as AIM

    from src.graph.checkpoint_serde import get_checkpoint_serde
    from src.obs import llm_invoke as li
    from src.reverie.cancel import (
        begin_reverie_cycle,
        end_reverie_cycle,
        reset_reverie_cancel_state,
    )
    from src.reverie.turn_kind import TurnKind, turn_kind

    reset_reverie_cancel_state()

    class _StreamLLM:
        def invoke(self, *_a, **_k):
            raise AssertionError("must stream")

        def stream(self, _msgs, config=None):
            yield AIMessageChunk(
                content="",
                tool_call_chunks=[
                    {
                        "name": "search_lessons",
                        "args": '{"query":',
                        "id": "tc1",
                        "index": 0,
                    }
                ],
                additional_kwargs={"reasoning_content": "internal plan"},
            )
            yield AIMessageChunk(
                content="",
                tool_call_chunks=[
                    {
                        "name": None,
                        "args": '"alpha"}',
                        "id": None,
                        "index": 0,
                    }
                ],
            )

    token = begin_reverie_cycle()
    try:
        with turn_kind(TurnKind.SOLITARY):
            out = li.invoke_llm(
                _StreamLLM(), [{"role": "user", "content": "x"}], role="agent"
            )
    finally:
        end_reverie_cycle(token)
        reset_reverie_cancel_state()

    assert type(out) is AIM
    assert not isinstance(out, AIMessageChunk)
    assert out.tool_calls
    assert out.tool_calls[0]["name"] == "search_lessons"
    assert isinstance(out.tool_calls[0]["args"], dict)
    assert out.tool_calls[0]["args"].get("query") == "alpha"
    assert (out.additional_kwargs or {}).get("reasoning_content") == "internal plan"

    serde = get_checkpoint_serde()
    blob = serde.dumps_typed(out)
    restored = serde.loads_typed(blob)
    assert type(restored) is AIM
    assert restored.tool_calls[0]["args"].get("query") == "alpha"


@pytest.mark.unit
def test_live_path_uses_invoke_not_stream():
    from src.obs import llm_invoke as li

    class _InvokeLLM:
        def invoke(self, *_a, **_k):
            return AIMessage(content="live")

        def stream(self, *_a, **_k):
            raise AssertionError("live must not stream")

    # No turn_kind → LIVE → invoke path.
    out = li.invoke_llm(_InvokeLLM(), [{"role": "user", "content": "x"}], role="agent")
    assert out.content == "live"


@pytest.mark.unit
def test_resume_saved_on_cancel_and_cleared_on_ran(monkeypatch, reset_solitary_state):
    from src.agent_runner import AgentTurnResult
    from src.config import settings
    from src.pulse.rhythm import Verdict
    from src.reverie import solitary as sol
    from src.reverie.background_deliver import BackgroundDeliverResult

    monkeypatch.setattr(settings, "solitary_enabled", True, raising=False)
    monkeypatch.setattr(settings, "solitary_interval_min", 0, raising=False)
    monkeypatch.setattr(sol, "may", lambda *_a, **_k: Verdict(allowed=True))
    monkeypatch.setattr(sol, "mark_solitary_ran", lambda: None)
    monkeypatch.setattr(sol, "_load_solitary_prompt", lambda: "seed")
    monkeypatch.setattr(
        "src.policy_knobs.solitary_deferred_for_live_busy",
        lambda *a, **k: False,
    )
    monkeypatch.setattr(sol, "_thread_has_checkpoint", lambda _tid: True)

    sol._resume_thread_id = None
    phase = {"n": 0}
    captured: list[dict] = []

    def _turn(message, *, thread_id, session_uuid, diary_source=None):
        captured.append({"message": message, "thread_id": thread_id})
        return AgentTurnResult(reply="ok", tools_used=(), elapsed_ms=1.0)

    monkeypatch.setattr("src.agent_runner.run_agent_turn_detailed", _turn)

    def _deliver(*, kind, run_turn, **_kw):
        phase["n"] += 1
        if phase["n"] == 1:
            return BackgroundDeliverResult(
                result="cancelled",
                reply="",
                turn=None,
                model_queued=False,
                fallback_queued=False,
            )
        turn = run_turn()
        return BackgroundDeliverResult(
            result="ran",
            reply=getattr(turn, "reply", "") or "",
            turn=turn,
            model_queued=False,
            fallback_queued=False,
        )

    monkeypatch.setattr(sol, "run_background_turn_and_deliver", _deliver)

    assert sol.run_solitary_pass_if_due() == "cancelled"
    assert sol._resume_thread_id is not None
    saved = sol._resume_thread_id

    assert sol.run_solitary_pass_if_due() == "ran"
    assert len(captured) == 1
    assert captured[0]["thread_id"] == saved
    assert "interrupted by a live USER turn" in captured[0]["message"]
    assert sol._resume_thread_id is None

    # Gate-reject (no checkpoint) must not arm resume.
    sol._resume_thread_id = None
    phase["n"] = 0
    monkeypatch.setattr(sol, "_thread_has_checkpoint", lambda _tid: False)

    def _deliver_cancel_only(*, kind, run_turn, **_kw):
        return BackgroundDeliverResult(
            result="cancelled",
            reply="",
            turn=None,
            model_queued=False,
            fallback_queued=False,
        )

    monkeypatch.setattr(sol, "run_background_turn_and_deliver", _deliver_cancel_only)
    assert sol.run_solitary_pass_if_due() == "cancelled"
    assert sol._resume_thread_id is None
