"""Agent OTel metrics. Instruments bind after ``init_telemetry`` sets the provider."""

from __future__ import annotations

from opentelemetry.metrics import Counter, Histogram

from src.obs.tracing import get_meter, is_telemetry_enabled

# Populated by ``_bind_instruments()`` from ``init_telemetry`` (not at import time).
turn_duration: Histogram | None = None
tool_calls: Counter | None = None
tokens_in: Counter | None = None
tokens_out: Counter | None = None
emergency_events: Counter | None = None
junk_gate_rejections: Counter | None = None
reply_extraction_fallback: Counter | None = None
dpo_gold_insert_failures: Counter | None = None
reverie_passes: Counter | None = None
pulse_intent_runs: Counter | None = None
memory_contradiction_events: Counter | None = None
retrieve_distrust_applied: Counter | None = None
write_contamination: Counter | None = None
write_ungrounded_reject: Counter | None = None
llm_length_cut: Counter | None = None


def _bind_instruments() -> None:
    """Create instruments on the active MeterProvider (call once after OTel init)."""
    global turn_duration, tool_calls, tokens_in, tokens_out, emergency_events, junk_gate_rejections, reply_extraction_fallback, dpo_gold_insert_failures, reverie_passes, pulse_intent_runs, memory_contradiction_events, retrieve_distrust_applied, write_contamination, write_ungrounded_reject, llm_length_cut
    meter = get_meter("agent")
    turn_duration = meter.create_histogram(
        "agent.turn.duration",
        unit="s",
        description="Full turn latency",
    )
    tool_calls = meter.create_counter(
        "agent.tool.calls",
        description="Tool invocations",
        unit="1",
    )
    tokens_in = meter.create_counter("agent.llm.tokens.input", unit="1")
    tokens_out = meter.create_counter("agent.llm.tokens.output", unit="1")
    emergency_events = meter.create_counter("agent.emergency.events", unit="1")
    junk_gate_rejections = meter.create_counter("agent.junk_gate.rejections", unit="1")
    reply_extraction_fallback = meter.create_counter(
        "agent.reply.extraction_fallback",
        description="Reply visible-text extraction fallbacks (abliterated / template drift)",
        unit="1",
    )
    dpo_gold_insert_failures = meter.create_counter(
        "agent.dpo.gold_insert_failures",
        description="Failed DPO gold pair inserts (missing table / constraint)",
        unit="1",
    )
    reverie_passes = meter.create_counter(
        "agent.reverie.passes",
        description="Solitary reverie cycle outcomes",
        unit="1",
    )
    pulse_intent_runs = meter.create_counter(
        "agent.pulse.intent_runs",
        description="Pulse intent pass outcomes",
        unit="1",
    )
    memory_contradiction_events = meter.create_counter(
        "agent.memory.contradiction_events",
        description="Memory/diary contradiction gate outcomes (policy + resolution)",
        unit="1",
    )
    retrieve_distrust_applied = meter.create_counter(
        "agent.retrieve.distrust_applied",
        description="[MEM] candidates downranked by retrieve distrust",
        unit="1",
    )
    write_contamination = meter.create_counter(
        "agent.write.contamination",
        description="Background writes with invented USER: prefix",
        unit="1",
    )
    write_ungrounded_reject = meter.create_counter(
        "agent.write.ungrounded_reject",
        description="Background writes rejected by lexical/id evidence filter",
        unit="1",
    )
    llm_length_cut = meter.create_counter(
        "agent.llm.length_cut",
        description="LLM generations that finished on done_reason=length",
        unit="1",
    )


def _reset_instruments_for_tests() -> None:
    """Clear instrument handles (unit tests only)."""
    global turn_duration, tool_calls, tokens_in, tokens_out, emergency_events, junk_gate_rejections, reply_extraction_fallback, dpo_gold_insert_failures, reverie_passes, pulse_intent_runs, memory_contradiction_events, retrieve_distrust_applied, write_contamination, write_ungrounded_reject, llm_length_cut
    turn_duration = None
    tool_calls = None
    tokens_in = None
    tokens_out = None
    emergency_events = None
    junk_gate_rejections = None
    reply_extraction_fallback = None
    dpo_gold_insert_failures = None
    reverie_passes = None
    pulse_intent_runs = None
    memory_contradiction_events = None
    retrieve_distrust_applied = None
    write_contamination = None
    write_ungrounded_reject = None
    llm_length_cut = None


def record_turn_duration(
    seconds: float,
    channel: str,
    *,
    concurrent_deferred_job: bool = False,
) -> None:
    if not is_telemetry_enabled() or turn_duration is None:
        return
    turn_duration.record(
        max(0.0, float(seconds)),
        {
            "channel": channel or "unknown",
            "concurrent_deferred_job": ("true" if concurrent_deferred_job else "false"),
        },
    )


def record_tool_call(tool_name: str) -> None:
    if not is_telemetry_enabled() or tool_calls is None:
        return
    tool_calls.add(1, {"tool": tool_name or "unknown"})


def record_llm_tokens(output_tokens: int | None, input_tokens: int | None) -> None:
    if not is_telemetry_enabled() or tokens_in is None or tokens_out is None:
        return
    if input_tokens is not None and input_tokens > 0:
        tokens_in.add(int(input_tokens))
    if output_tokens is not None and output_tokens > 0:
        tokens_out.add(int(output_tokens))


def record_emergency_event(reason: str) -> None:
    if not is_telemetry_enabled() or emergency_events is None:
        return
    emergency_events.add(1, {"reason": reason or "unknown"})


def record_junk_gate_rejection(reason: str) -> None:
    if not is_telemetry_enabled() or junk_gate_rejections is None:
        return
    junk_gate_rejections.add(1, {"reason": reason or "unknown"})


def record_reply_extraction_fallback(reason: str) -> None:
    if not is_telemetry_enabled() or reply_extraction_fallback is None:
        return
    reply_extraction_fallback.add(1, {"reason": reason or "unknown"})


def record_dpo_gold_insert_failure() -> None:
    if not is_telemetry_enabled() or dpo_gold_insert_failures is None:
        return
    dpo_gold_insert_failures.add(1)


def record_reverie_pass(result: str) -> None:
    if not is_telemetry_enabled() or reverie_passes is None:
        return
    reverie_passes.add(1, {"result": result or "unknown"})


def record_pulse_intent_run(result: str) -> None:
    if not is_telemetry_enabled() or pulse_intent_runs is None:
        return
    pulse_intent_runs.add(1, {"result": result or "unknown"})


def record_memory_contradiction_event(*, policy: str, resolution: str) -> None:
    """``resolution``: demoted | replace | unverified | skip_unverified | log_pile | seen."""
    if not is_telemetry_enabled() or memory_contradiction_events is None:
        return
    memory_contradiction_events.add(
        1,
        {
            "policy": (policy or "unknown")[:32],
            "resolution": (resolution or "unknown")[:32],
        },
    )


def record_retrieve_distrust_applied(count: int = 1) -> None:
    if not is_telemetry_enabled() or retrieve_distrust_applied is None:
        return
    n = max(0, int(count))
    if n:
        retrieve_distrust_applied.add(n)


def record_write_contamination(reason: str) -> None:
    if not is_telemetry_enabled() or write_contamination is None:
        return
    write_contamination.add(1, {"reason": (reason or "unknown")[:32]})


def record_write_ungrounded_reject(reason: str) -> None:
    if not is_telemetry_enabled() or write_ungrounded_reject is None:
        return
    write_ungrounded_reject.add(1, {"reason": (reason or "unknown")[:32]})


def record_llm_length_cut() -> None:
    if not is_telemetry_enabled() or llm_length_cut is None:
        return
    llm_length_cut.add(1)


__all__ = [
    "_bind_instruments",
    "_reset_instruments_for_tests",
    "dpo_gold_insert_failures",
    "emergency_events",
    "junk_gate_rejections",
    "memory_contradiction_events",
    "record_dpo_gold_insert_failure",
    "record_emergency_event",
    "record_junk_gate_rejection",
    "llm_length_cut",
    "record_llm_length_cut",
    "record_llm_tokens",
    "record_memory_contradiction_event",
    "record_reply_extraction_fallback",
    "record_retrieve_distrust_applied",
    "record_reverie_pass",
    "record_pulse_intent_run",
    "record_tool_call",
    "record_turn_duration",
    "record_write_contamination",
    "record_write_ungrounded_reject",
    "reply_extraction_fallback",
    "write_contamination",
    "write_ungrounded_reject",
    "retrieve_distrust_applied",
    "reverie_passes",
    "pulse_intent_runs",
    "tokens_in",
    "tokens_out",
    "tool_calls",
    "turn_duration",
]
