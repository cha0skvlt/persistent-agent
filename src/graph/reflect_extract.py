"""Reflect extraction — world facts + diary (split from workflow god-module)."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.constants import TAG_NOSTREAM

from src.config import settings
from src.graph.diary_echo_gate import skip_diary_echo_duplicate
from src.graph.junk_gate import is_junk
from src.graph.reflect_consistency import gate_diary_write, gate_memory_write
from src.graph.turn_evidence import live_extract_evidence
from src.memory import get_default_memory, get_default_self_reflection
from src.memory_labels import world_fact_label_error
from src.obs.logs import is_debug_enabled, log_debug_event, log_event
from src.obs.metrics import record_junk_gate_rejection
from src.obs.spans import traced_node
from src.reply.visible import strip_redacted_thinking
from src.reverie.distill import gate_background_provenance
from src.speaker import speaker_label, subject_is_agent
from src.state.agent_state import AgentState
from src.utils import get_message_text, human_message_text, is_synthetic_human_text

_log = logging.getLogger(__name__)

# solitary plus legacy diary_source still present in older DB rows
_BACKGROUND_DIARY_SOURCES = frozenset({"solitary", "vigil"})
# Background diary writes that must go through distill (includes intent).
_DISTILL_DIARY_SOURCES = frozenset({"solitary", "vigil", "intent"})


def _parse_reflect_json(raw: Any) -> dict[str, Any]:
    """First valid JSON object in the model output.

    ``json.loads`` on the whole string first, then ``raw_decode`` from every
    ``{``. A greedy ``\\{[\\s\\S]*\\}`` regex swallows trailing prose and
    multiple objects, and loses the one that actually parses.
    """
    if not isinstance(raw, str):
        return {}
    text = raw.strip()
    if not text:
        return {}
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    decoder = json.JSONDecoder()
    for pos, char in enumerate(text):
        if char != "{":
            continue
        try:
            data, _ = decoder.raw_decode(text[pos:])
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(data, dict):
            return data
    return {}


_MAX_FACTS = 20
_MAX_FACT_CHARS = 480
_MAX_REFLECTION_CHARS = 280


def _valid_facts(data: dict[str, Any]) -> list[str]:
    """Schema check for the world-facts payload: bounded list of bounded strings."""
    raw = data.get("facts")
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for item in raw[:_MAX_FACTS]:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if not text:
            continue
        out.append(text[:_MAX_FACT_CHARS])
    return out


def _valid_reflection(data: dict[str, Any]) -> str | None:
    """Schema check for the diary payload: one bounded string or None."""
    raw = data.get("reflection")
    if raw is None or not isinstance(raw, str):
        return None
    text = raw.strip()
    if not text:
        return None
    return text[:_MAX_REFLECTION_CHARS]


def _invoke_reflect_llm(system: str, user_prompt: str, *, path: str) -> str:
    from src import llm_factory
    from src.obs.debug_context import emit_llm_context
    from src.obs.llm_invoke import invoke_llm

    messages = [
        SystemMessage(content=system),
        HumanMessage(content=user_prompt),
    ]
    if is_debug_enabled():
        emit_llm_context(path, system, messages, "")
    res = invoke_llm(
        llm_factory.get_llm(),
        messages,
        role=path,
        config={"tags": [TAG_NOSTREAM]},
    )
    raw = getattr(res, "content", "")
    if is_debug_enabled():
        log_debug_event("llm.reflect_raw", text=str(raw or ""), path=path)
    return str(raw or "")


_SUBJECT_LEAD_SKIP = frozenset(
    {
        "the",
        "a",
        "an",
        "this",
        "that",
        "currently",
        "now",
        "also",
        "moreover",
        "recently",
        "today",
        "still",
        "generally",
        "typically",
    }
)

_SUBJECT_TOKEN_RE = re.compile(r"[A-Za-z']+")


def _third_person_agent_fact(fact: str) -> bool:
    """True when a world-fact line names the agent in third person without AGENT: prefix.

    Locates the grammatical subject past leading articles/adverbs and strips a
    trailing possessive, so "The agent…", "The assistant's memory…", "Currently
    <self-anchor> is…" are rejected — not only bare self-anchor / "agent …".
    """
    clean = (fact or "").strip()
    if not clean:
        return False
    upper = clean.upper()
    if upper.startswith("AGENT:") or upper.startswith("USER:"):
        return False

    tokens = _SUBJECT_TOKEN_RE.findall(clean)
    if not tokens:
        return False
    lowered = [tok.casefold() for tok in tokens]

    idx = 0
    while idx < len(lowered) and lowered[idx] in _SUBJECT_LEAD_SKIP:
        idx += 1
    if idx >= len(lowered):
        return False

    subject = lowered[idx]
    if subject.endswith("'s"):
        subject = subject[:-2]
    elif subject.endswith("'"):
        subject = subject[:-1]
    subject = subject.strip("'")

    return subject_is_agent(subject)


def _run_world_facts_reflect(
    user_text: str,
    agent_text: str,
    mem: Any,
    *,
    turn_count: int,
    deferred: bool = False,
    messages: list[Any] | None = None,
    evidence: str | None = None,
) -> None:
    every = int(settings.reflect_every_n)
    if every <= 0 or turn_count % every != 0:
        return

    anchor = speaker_label()
    extract_evidence = (
        evidence if evidence is not None else live_extract_evidence(user_text, messages)
    )
    prompt = (
        "EXTRACT facts only from USER and EVIDENCE (this turn's tools).\n"
        "facts: a list of standalone world facts (USER, projects, hardware, "
        f"configs, stack). NOT facts about {anchor}/Agent in third person.\n"
        "Prefix USER: … for facts about the person; neutral world facts — no prefix; "
        'never first person "I" — that is the diary, not world memory.\n'
        "Do not use the agent's reply as a source.\n\n"
        'Return STRICT JSON: {"facts":["..."]}\n\n'
        f"USER: {user_text}\n"
        f"EVIDENCE:\n{extract_evidence}\n"
    )
    facts_saved = 0
    try:
        raw = _invoke_reflect_llm(
            "Return valid JSON only.", prompt, path="reflect_world"
        )
        data = _parse_reflect_json(raw)
        if (raw or "").strip() and not data:
            log_event("reflect.parse_failed", preview=str(raw)[:120])
        facts = _valid_facts(data)

        gate_enabled = bool(settings.reflect_consistency_gate_enabled and deferred)

        for fact in facts:
            if not (isinstance(fact, str) and fact.strip()):
                continue
            clean = fact.strip()
            if _third_person_agent_fact(clean):
                log_event(
                    "reflect.blocked",
                    reason="third_person_agent",
                    preview=clean[:40],
                )
                continue
            if is_junk(clean, user_text=user_text, agent_text=agent_text):
                record_junk_gate_rejection("junk_gate")
                log_event(
                    "reflect.blocked",
                    reason="junk_gate",
                    preview=clean[:40],
                )
                continue
            if _is_junk_fact(clean):
                log_event(
                    "reflect.blocked",
                    reason="code_fragment",
                    preview=clean[:40],
                )
                continue
            label_err = world_fact_label_error(clean)
            if label_err == "prefix_required":
                log_event(
                    "reflect.blocked",
                    reason="world_fact_deixis",
                    preview=clean[:40],
                )
                continue
            blocked = gate_background_provenance(clean, evidence=extract_evidence)
            if blocked:
                log_event(
                    "reflect.blocked",
                    reason="no_evidence",
                    preview=clean[:40],
                )
                continue
            if gate_enabled:
                if gate_memory_write(mem, clean):
                    facts_saved += 1
            else:
                fact_source = "user" if clean.upper().startswith("USER:") else "tool"
                if (
                    mem.save(
                        clean,
                        memory_type="semantic",
                        agent=settings.default_agent_id,
                        source=fact_source,
                        source_ref=f"turn:{turn_count}",
                    )
                    is not None
                ):
                    facts_saved += 1

        log_event("reflect.extract", facts=facts_saved)
    except Exception as e:
        _log.warning("World-facts reflection skipped: %s", e)


def _run_self_reflect(
    user_text: str,
    agent_text: str,
    turn_count: int,
    *,
    deferred: bool = False,
    source: str = "reflect",
) -> None:
    every = int(settings.self_reflect_every_n)
    if every <= 0:
        return
    if source not in _BACKGROUND_DIARY_SOURCES and turn_count % every != 0:
        return

    anchor = speaker_label()
    speaker = speaker_label()
    prompt = (
        f"You are {anchor}. Look at the exchange — what did you notice about yourself?\n"
        "1-2 sentences. First person, present tense, in English.\n"
        f'NOT "{anchor} — …" — write "I". A diary entry.\n'
        'Return STRICT JSON: {"reflection": "..."} or {"reflection": null}.\n\n'
        f"USER: {user_text}\n"
        f"{speaker}: {agent_text}\n"
    )
    try:
        raw = _invoke_reflect_llm(
            "Return valid JSON only.",
            prompt,
            path="reflect_self",
        )
        data = _parse_reflect_json(raw)
        text = _valid_reflection(data)
        if text is None:
            return
        if source in _DISTILL_DIARY_SOURCES:
            from src.reverie.distill import distill_solitary_diary

            text = distill_solitary_diary(text) or ""
            if not text:
                log_event("reflect.self_skip", reason="distill_rejected", source=source)
                return
        sr = get_default_self_reflection()
        if deferred and skip_diary_echo_duplicate(sr, text):
            return
        if bool(settings.reflect_consistency_gate_enabled and deferred):
            if gate_diary_write(sr, text, source=source):
                log_event("reflect.self", preview=text[:60], source=source)
        elif sr.save(text, source=source):
            log_event("reflect.self", preview=text[:60], source=source)
    except Exception as e:
        _log.warning("Self-reflection skipped: %s", e)


# ---------------------------------------------------------------------------
# Reflect quality gate — block code fragments and technical junk from durable memory
# ---------------------------------------------------------------------------

_JUNK_CODE_LINE_RE = re.compile(
    r"^\s*(import |from \w+ import|def \w+\s*\(|class \w+[\(:]|#!/|>>>\s|\$\s)",
    re.MULTILINE,
)
_JUNK_RUNTIME_RE = re.compile(
    r"\bos\.system\s*\(|\bsubprocess\.\w+|\beval\s*\(|\bexec\s*\("
    r"|\bwhile\s+True\b|\bfor\s+\w+\s+in\s+"
)
# Bare key=value line like  ProxyCommand="localhost:2222"  or  PORT=8080
_JUNK_KV_LINE_RE = re.compile(r"^\w[\w-]*\s*=\s*[\"']?[\w./:\\-]+[\"']?\s*$")


def _is_junk_fact(fact: str) -> bool:
    """Return True when a reflect-extracted fact looks like code or technical junk."""
    t = fact.strip()
    if not t:
        return True
    if _JUNK_CODE_LINE_RE.search(t):
        return True
    if _JUNK_RUNTIME_RE.search(t):
        return True
    if _JUNK_KV_LINE_RE.match(t):
        return True
    # Multiline block where majority of lines after the first are indented → code block
    lines = [ln for ln in t.splitlines() if ln.strip()]
    if len(lines) > 1:
        indented = sum(1 for ln in lines[1:] if ln and ln[0] in " \t")
        if indented / len(lines[1:]) >= 0.5:
            return True
    # Must have at least one natural-language word (≥5 letters, any script)
    if not re.search(r"[^\W\d_]{5,}", t, flags=re.UNICODE):
        return True
    return False


def _find_last_human_ai_pair(
    msgs: list[Any],
) -> tuple[HumanMessage | None, AIMessage | None]:
    """Find the last (HumanMessage, AIMessage) pair; skip ToolMessages and tool-only AIMessages."""
    for i in range(len(msgs) - 1, 0, -1):
        if isinstance(msgs[i], AIMessage) and not msgs[i].tool_calls:
            for j in range(i - 1, -1, -1):
                if isinstance(msgs[j], HumanMessage):
                    if is_synthetic_human_text(human_message_text(msgs[j])):
                        continue
                    return msgs[j], msgs[i]
                # Same graph turn: agent may emit tool_calls before a final text reply.
                if isinstance(msgs[j], AIMessage) and not msgs[j].tool_calls:
                    break
    return None, None


def _skip_world_facts_reflect(diary_source: str) -> bool:
    """True when world-fact extract would write around the provenance filter."""
    if (diary_source or "").strip() in _DISTILL_DIARY_SOURCES:
        return True
    try:
        from src.reverie.turn_kind import is_background_turn

        return bool(is_background_turn())
    except Exception:
        return False


@traced_node("reflect")
def reflect_node(state: AgentState):
    """Extract durable facts from last user/agent turn and persist (reflection)."""
    log_event("node.start", node="reflect_node")
    t0 = time.perf_counter()
    try:
        if not settings.reflection:
            return {}
        # Abort only when BOTH world-facts and self-reflection are fully disabled.
        world_disabled = int(settings.reflect_every_n) <= 0
        self_disabled = int(settings.self_reflect_every_n) <= 0
        if world_disabled and self_disabled:
            return {}

        if not bool(state.get("needs_reflect", False)):
            return {}

        msgs = list(state.get("messages") or [])
        if len(msgs) < 2:
            return {}

        turn_count = int(state.get("turn_count", 0) or 0)
        every_n = int(settings.reflect_every_n)

        prev_msg, last_msg = _find_last_human_ai_pair(msgs)
        if prev_msg is None or last_msg is None:
            log_event("reflect.skip", reason="last_turn_not_user_agent_pair")
            return {}

        user_text = getattr(prev_msg, "content", "")
        if not isinstance(user_text, str):
            user_text = str(user_text)
        agent_text = strip_redacted_thinking(get_message_text(last_msg))
        log_event("reflect.run", turn_count=turn_count, reflect_every_n=every_n)
        log_debug_event(
            "reflect.pair",
            user_chars=len(user_text),
            agent_chars=len(agent_text),
        )

        mem = get_default_memory()
        deferred = bool(state.get("_reflect_deferred", False))
        diary_source = str(state.get("diary_source") or "reflect")
        # Background: world facts only via tools + provenance filter — not
        # a second-LLM dump into the store (intent included).
        if _skip_world_facts_reflect(diary_source):
            log_event(
                "reflect.world_skip",
                reason="solitary_distill_only",
                source=diary_source,
            )
        else:
            _run_world_facts_reflect(
                user_text,
                agent_text,
                mem,
                turn_count=turn_count,
                deferred=deferred,
                messages=msgs,
            )
        _run_self_reflect(
            user_text,
            agent_text,
            turn_count,
            deferred=deferred,
            source=diary_source,
        )

        if turn_count > 0:
            return {"last_reflect_turn": turn_count}
        return {}
    finally:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        log_event("node.timing", node="reflect_node", elapsed_ms=elapsed_ms)
