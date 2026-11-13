"""Recover tool calls emitted as plain JSON in ``AIMessage.content`` (model quirk)."""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any

from langchain_core.messages import AIMessage

_log = logging.getLogger(__name__)

HALLUCINATED_TOOL_PREFIX_RE = re.compile(r'^\{"name"\s*:\s*"([^"]+)"')

# Qwen-style XML tool dumps in content (not native tool_calls).
# One ``<function=…>…</function>`` body — params must not bleed into the next call.
_XML_FUNCTION_BLOCK_RE = re.compile(
    r"<function\s*=\s*([A-Za-z_][\w]*)\s*>"
    r"([\s\S]*?)"
    r"(?:</function\s*>|(?=<function\s*=)|$)",
    re.IGNORECASE,
)
_XML_PARAM_RE = re.compile(
    r"<parameter\s*=\s*([A-Za-z_][\w]*)\s*>"
    r"([\s\S]*?)"
    r"(?:</parameter\s*>|(?=<parameter\s*=)|(?=</function)|(?=</tool_call)|$)",
    re.IGNORECASE,
)

# Bare ``name(key="value", ...)`` calls: a model without native tool calling
# writes the call as prose. Only known tool names are considered.
_BARE_CALL_RE = re.compile(r"(?m)^\s*([a-z_][a-z0-9_]*)\s*\(\s*([^()]*?)\s*\)\s*$")
_BARE_KWARG_RE = re.compile(
    r"""([a-z_][a-z0-9_]*)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^,\s]+))"""
)


def try_materialize_htc_tool_call(
    response: AIMessage,
    *,
    allowed_tool_names: frozenset[str],
) -> AIMessage | None:
    """If ``content`` is a recoverable tool blob (JSON or XML), return structured AIMessage.

    Returns ``None`` when there is nothing to recover (already has tool_calls,
    unknown tool, or invalid args). Qwen may emit several ``<tool_call>`` blocks
    in one content string — all valid ones are materialized.
    """
    existing = getattr(response, "tool_calls", None)
    if isinstance(existing, list) and existing:
        return None
    raw = getattr(response, "content", "") or ""
    if not isinstance(raw, str) or not raw.strip():
        return None

    # JSON scan first: several {"name": …} blobs in one content string
    # (Qwen <tool_call> wrappers, trailing tags, think prose around).
    recovered_list = _iter_json_tool_blobs(raw, allowed_tool_names=allowed_tool_names)
    if not recovered_list:
        recovered_list = recover_xml_hallucinated_tool_calls(
            raw, allowed_tool_names=allowed_tool_names
        )
    if not recovered_list:
        recovered_list = recover_bare_call_tool_calls(
            raw, allowed_tool_names=allowed_tool_names
        )
    if not recovered_list:
        one = recover_hallucinated_tool_call(raw, allowed_tool_names=allowed_tool_names)
        if one is not None:
            recovered_list = [one]
    if not recovered_list:
        return None

    tool_calls: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for recovered in recovered_list:
        validated = validate_recovered_tool_args(recovered["name"], recovered["args"])
        if validated is None:
            continue
        # Model may dump the same call twice in one blob — execute once.
        key = (recovered["name"], repr(sorted(validated.items())))
        if key in seen:
            continue
        seen.add(key)
        tool_calls.append(
            {
                "name": recovered["name"],
                "args": validated,
                "id": f"htc_{uuid.uuid4().hex[:8]}",
                "type": "tool_call",
            }
        )
    if not tool_calls:
        return None
    return AIMessage(content="", tool_calls=tool_calls)


# Positional order for tools a model may call as ``name(value, ...)``.
_POSITIONAL_ARGS: dict[str, tuple[str, ...]] = {
    "save_memory": ("content",),
    "search_memory": ("query",),
    "search_diary": ("query",),
    "search_graph": ("query",),
    "search_web": ("query",),
    "search_lessons": ("query",),
    "fetch_url": ("url",),
    "find_related": ("entity",),
    "forget_memory": ("memory_id",),
    "forget_diary": ("reflection_id",),
    "cancel_intent": ("intent_id",),
    "message_user": ("text",),
    "record_lesson": ("change", "because"),
}


def _split_call_args(s: str) -> list[str]:
    """Split ``a, b="x, y", c`` on top-level commas only."""
    out: list[str] = []
    buf: list[str] = []
    depth = 0
    quote: str | None = None
    for ch in s or "":
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in "\"'":
            quote = ch
            buf.append(ch)
            continue
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(buf).strip())
            buf = []
            continue
        buf.append(ch)
    if buf:
        out.append("".join(buf).strip())
    return [x for x in out if x]


def _unquote(token: str) -> str:
    t = token.strip()
    if len(t) >= 2 and t[0] == t[-1] and t[0] in "\"'":
        return t[1:-1]
    return t


def recover_bare_call_tool_calls(
    text: str, *, allowed_tool_names: frozenset[str]
) -> list[dict]:
    """Parse ``name(key="value", ...)`` and ``name(value, ...)`` lines."""
    out: list[dict] = []
    for m in _BARE_CALL_RE.finditer(text or ""):
        name = m.group(1)
        if name not in allowed_tool_names:
            continue
        args: dict[str, str] = {}
        positional: list[str] = []
        for token in _split_call_args(m.group(2) or ""):
            km = re.match(r"^([a-z_][a-z0-9_]*)\s*=\s*(.+)$", token, re.S)
            if km:
                args[km.group(1)] = _unquote(km.group(2))
            else:
                positional.append(_unquote(token))
        slots = _POSITIONAL_ARGS.get(name, ())
        # ``save_memory(content, "value")``: the model names the parameter as a
        # bare token, then passes the value. Drop the redundant name.
        if positional and slots and positional[0] in slots:
            positional = positional[1:]
        for slot, value in zip(slots, positional, strict=False):
            args.setdefault(slot, value)
        out.append({"name": name, "args": args})
    return out


def recover_xml_hallucinated_tool_calls(
    text: str, *, allowed_tool_names: frozenset[str]
) -> list[dict]:
    """Parse every ``<function=name><parameter=…>`` block in model content."""
    out: list[dict] = []
    for m in _XML_FUNCTION_BLOCK_RE.finditer(text or ""):
        name = m.group(1)
        if name not in allowed_tool_names:
            continue
        args: dict[str, str] = {}
        for pm in _XML_PARAM_RE.finditer(m.group(2) or ""):
            key = pm.group(1)
            val = (pm.group(2) or "").strip()
            if key and val:
                args[key] = val
        out.append({"name": name, "args": args})
    return out


def recover_xml_hallucinated_tool_call(
    text: str, *, allowed_tool_names: frozenset[str]
) -> dict | None:
    """Parse the first ``<function=name><parameter=k>v`` dump from model content."""
    items = recover_xml_hallucinated_tool_calls(
        text, allowed_tool_names=allowed_tool_names
    )
    return items[0] if items else None


def _iter_json_tool_blobs(
    text: str, *, allowed_tool_names: frozenset[str]
) -> list[dict]:
    """All ``{"name": …}`` objects in ``text`` via raw_decode — tolerates
    surrounding prose and trailing ``</tool_call>`` garbage; keeps order."""
    decoder = json.JSONDecoder()
    out: list[dict] = []
    i = 0
    raw = text or ""
    while True:
        brace = raw.find("{", i)
        if brace < 0:
            break
        if '"name"' not in raw[brace : brace + 120]:
            i = brace + 1
            continue
        try:
            obj, consumed = decoder.raw_decode(raw[brace:])
        except ValueError:
            i = brace + 1
            continue
        i = brace + max(1, consumed)
        if not isinstance(obj, dict):
            continue
        name = obj.get("name")
        args = obj.get("parameters") or obj.get("args") or obj.get("arguments") or {}
        if isinstance(name, str) and name in allowed_tool_names:
            out.append({"name": name, "args": args if isinstance(args, dict) else {}})
    return out


def recover_hallucinated_tool_call(
    text: str, *, allowed_tool_names: frozenset[str]
) -> dict | None:
    """Parse a tool-call blob from model ``content``. ``name`` must be in ``allowed_tool_names``.

    Supports JSON ``{"name":...}`` (optionally after thinking) and Qwen XML
    ``<tool_call><function=name><parameter=...>``.
    """
    stripped = (text or "").strip()
    if not stripped:
        return None

    # Prefer a JSON object if present (may follow </think>).
    json_candidate = stripped
    if not json_candidate.startswith("{"):
        brace = json_candidate.find("{")
        if brace >= 0 and '"name"' in json_candidate[brace : brace + 80]:
            json_candidate = json_candidate[brace:].strip()
        else:
            json_candidate = ""

    if json_candidate.startswith("{"):
        try:
            # raw_decode: tolerate trailing garbage («…}</tool_call>», prose).
            data, _consumed = json.JSONDecoder().raw_decode(json_candidate)
            if isinstance(data, dict):
                name = data.get("name")
                args = (
                    data.get("parameters")
                    or data.get("args")
                    or data.get("arguments")
                    or {}
                )
                if isinstance(name, str) and name in allowed_tool_names:
                    return {
                        "name": name,
                        "args": args if isinstance(args, dict) else {},
                    }
        except (json.JSONDecodeError, ValueError):
            pass
        m = HALLUCINATED_TOOL_PREFIX_RE.match(json_candidate)
        if m:
            name = m.group(1)
            if name not in allowed_tool_names:
                return None
            if name == "save_memory":
                cm = re.search(
                    r'"content"\s*:\s*"((?:[^"\\]|\\.){20,})', json_candidate
                )
                if cm:
                    return {"name": name, "args": {"content": cm.group(1)}}
            qm = re.search(
                r'"(?:query|subject|fact|s)"\s*:\s*"([^"]{3,})"', json_candidate
            )
            if qm:
                key = re.search(r'"(query|subject|fact|s)"\s*:', json_candidate)
                k = key.group(1) if key else "query"
                return {"name": name, "args": {k: qm.group(1)}}

    return recover_xml_hallucinated_tool_call(
        stripped, allowed_tool_names=allowed_tool_names
    )


def _query_arg(raw_args: dict, *, max_len: int = 2000) -> str | None:
    """Models write ``q=`` as often as ``query=``; accept both."""
    return _as_nonempty_str(raw_args.get("query") or raw_args.get("q"), max_len=max_len)


def _as_nonempty_str(val: Any, *, max_len: int) -> str | None:
    if not isinstance(val, str):
        return None
    s = val.strip()
    if not s:
        return None
    if len(s) > max_len:
        s = s[:max_len]
    return s


def validate_recovered_tool_args(name: str, raw_args: Any) -> dict[str, Any] | None:
    """Validate and normalize args for a recovered tool call before ToolNode runs."""
    if not isinstance(raw_args, dict):
        _log.warning(
            "Hallucinated tool-call recovery rejected tool=%r: args must be object, got %s",
            name,
            type(raw_args).__name__,
        )
        return None

    def reject(reason: str) -> None:
        _log.warning(
            "Hallucinated tool-call recovery rejected tool=%r: %s args=%r",
            name,
            reason,
            raw_args,
        )

    if name == "search_memory":
        q = _query_arg(raw_args)
        if not q:
            reject("missing or empty query")
            return None
        return {"query": q}

    if name == "save_memory":
        # Models write ``text=`` as often as ``content=``; accept both.
        c = _as_nonempty_str(
            raw_args.get("content") or raw_args.get("text"), max_len=4000
        )
        if not c:
            reject("missing or empty content")
            return None
        return {"content": c}

    if name == "save_fact_to_graph":
        subj = _as_nonempty_str(raw_args.get("subject"), max_len=500)
        pred = _as_nonempty_str(raw_args.get("predicate"), max_len=500)
        obj = raw_args.get("object")
        if obj is None:
            obj = raw_args.get("object_str")
        obj_s = _as_nonempty_str(obj, max_len=500)
        if not subj or not pred or not obj_s:
            reject("subject/predicate/object must all be non-empty strings")
            return None
        return {"subject": subj, "predicate": pred, "object": obj_s}

    if name == "add_triple":
        subj = _as_nonempty_str(raw_args.get("subject"), max_len=500)
        pred = _as_nonempty_str(raw_args.get("predicate"), max_len=500)
        oent = raw_args.get("object_entity")
        if oent is None:
            oent = raw_args.get("object")
        oent_s = _as_nonempty_str(oent, max_len=500)
        if not subj or not pred or not oent_s:
            reject("subject/predicate/object_entity must all be non-empty strings")
            return None
        return {"subject": subj, "predicate": pred, "object_entity": oent_s}

    if name == "search_graph":
        q = _query_arg(raw_args)
        if not q:
            reject("missing or empty query")
            return None
        return {"query": q}

    if name == "find_related":
        n = _as_nonempty_str(raw_args.get("name"), max_len=500)
        if not n:
            reject("missing or empty name")
            return None
        out: dict[str, Any] = {"name": n}
        if "depth" in raw_args:
            d = raw_args["depth"]
            try:
                di = int(d)
            except (TypeError, ValueError):
                reject("depth must be int-compatible")
                return None
            if di < 1 or di > 3:
                reject("depth out of range (1..3)")
                return None
            out["depth"] = di
        preds = raw_args.get("predicates")
        if preds is not None:
            if not isinstance(preds, list) or not all(
                isinstance(x, str) for x in preds
            ):
                reject("predicates must be a list of strings or omitted")
                return None
            out["predicates"] = [
                p.strip() for p in preds if isinstance(p, str) and p.strip()
            ]
        return out

    if name == "query_graph":
        subj = raw_args.get("subject")
        pred = raw_args.get("predicate")
        obj = raw_args.get("object_str")
        if subj is not None and not isinstance(subj, str):
            reject("subject must be string or null")
            return None
        if pred is not None and not isinstance(pred, str):
            reject("predicate must be string or null")
            return None
        if obj is not None and not isinstance(obj, str):
            reject("object_str must be string or null")
            return None
        subj_s = subj.strip() if isinstance(subj, str) else None
        pred_s = pred.strip() if isinstance(pred, str) else None
        obj_s = obj.strip() if isinstance(obj, str) else None
        if not subj_s and not pred_s and not obj_s:
            reject("at least one of subject/predicate/object_str must be non-empty")
            return None
        return {
            "subject": subj_s,
            "predicate": pred_s,
            "object_str": obj_s,
        }

    if name == "search_web":
        q = _query_arg(raw_args, max_len=500)
        if not q or len(q) < 2:
            reject("missing or too-short query")
            return None
        return {"query": q}

    if name == "fetch_url":
        u = _as_nonempty_str(raw_args.get("url"), max_len=2000)
        if not u:
            reject("missing or empty url")
            return None
        return {"url": u}

    if name == "browse_web":
        act = _as_nonempty_str(raw_args.get("action"), max_len=32)
        if not act:
            reject("missing or empty action")
            return None
        act_l = act.strip().lower()
        allowed = {"open", "follow", "click", "extract", "close"}
        if act_l not in allowed:
            reject(f"action must be one of {sorted(allowed)}")
            return None
        tgt = raw_args.get("target")
        tgt_s = ""
        if isinstance(tgt, str):
            tgt_s = tgt.strip()
        elif tgt is not None:
            tgt_s = str(tgt).strip()
        if act_l in {"open", "follow", "click"} and not tgt_s:
            reject(f"action={act_l} requires non-empty target")
            return None
        browse_args: dict[str, str] = {"action": act_l}
        if tgt_s:
            browse_args["target"] = tgt_s[:2000]
        return browse_args

    if name in {"forget_memory", "memory_replace"}:
        mid = _as_nonempty_str(raw_args.get("memory_id"), max_len=64)
        if not mid:
            reject("missing or empty memory_id")
            return None
        if name == "forget_memory":
            return {"memory_id": mid}
        body = _as_nonempty_str(
            raw_args.get("new_content") or raw_args.get("content"), max_len=4000
        )
        if not body:
            reject("missing or empty new_content")
            return None
        return {"memory_id": mid, "new_content": body}

    if name in {"forget_diary", "diary_replace"}:
        rid = _as_nonempty_str(
            raw_args.get("reflection_id") or raw_args.get("id"), max_len=64
        )
        if not rid:
            reject("missing or empty reflection_id")
            return None
        if name == "forget_diary":
            return {"reflection_id": rid}
        body = _as_nonempty_str(
            raw_args.get("new_content") or raw_args.get("content"), max_len=4000
        )
        if not body:
            reject("missing or empty new_content")
            return None
        return {"reflection_id": rid, "new_content": body}

    if name == "search_diary":
        q = _query_arg(raw_args)
        if not q:
            reject("missing or empty query")
            return None
        return {"query": q}

    if name == "run_selftest":
        return {}

    if name == "message_user":
        text = _as_nonempty_str(raw_args.get("text"), max_len=8000)
        if not text:
            reject("missing or empty text")
            return None
        return {"text": text}

    if name == "schedule_intent":
        text = _as_nonempty_str(raw_args.get("text"), max_len=4000)
        when = _as_nonempty_str(raw_args.get("when"), max_len=500)
        if not text or not when:
            reject("text and when must be non-empty strings")
            return None
        return {"text": text, "when": when}

    if name == "list_intents":
        return {}

    if name == "cancel_intent":
        iid = _as_nonempty_str(raw_args.get("intent_id"), max_len=64)
        if not iid:
            reject("missing or empty intent_id")
            return None
        return {"intent_id": iid}

    if name == "search_by_time":
        mode = _as_nonempty_str(raw_args.get("mode"), max_len=32) or "range"
        mode_l = mode.strip().lower()
        if mode_l not in {"range", "cadence", "evolution"}:
            reject("mode must be range|cadence|evolution")
            return None
        out_time: dict[str, Any] = {"mode": mode_l}
        for key, maxlen in (
            ("query", 2000),
            ("since", 64),
            ("until", 64),
            ("source", 32),
        ):
            if key not in raw_args:
                continue
            val = raw_args.get(key)
            if val is None or val == "":
                continue
            s = _as_nonempty_str(val, max_len=maxlen)
            if not s:
                reject(f"{key} must be a non-empty string when provided")
                return None
            out_time[key] = s
        return out_time

    if name == "record_lesson":
        change = _as_nonempty_str(raw_args.get("change"), max_len=2000)
        because = _as_nonempty_str(raw_args.get("because"), max_len=2000)
        if not change or not because:
            reject("change and because must be non-empty strings")
            return None
        out_lesson: dict[str, Any] = {"change": change, "because": because}
        if "evidence_ids" in raw_args and raw_args.get("evidence_ids") not in (
            None,
            "",
        ):
            ev = _as_nonempty_str(raw_args.get("evidence_ids"), max_len=2000)
            if not ev:
                reject("evidence_ids must be a non-empty string when provided")
                return None
            out_lesson["evidence_ids"] = ev
        return out_lesson

    if name == "search_lessons":
        q = _query_arg(raw_args)
        if not q:
            reject("missing or empty query")
            return None
        out_search: dict[str, Any] = {"query": q}
        if "limit" in raw_args and raw_args.get("limit") not in (None, ""):
            try:
                lim = int(raw_args.get("limit"))
            except (TypeError, ValueError):
                reject("limit must be an int")
                return None
            out_search["limit"] = max(1, min(8, lim))
        return out_search

    _log.warning(
        "Hallucinated tool-call recovery rejected tool=%r: no validator (add branch)",
        name,
    )
    return None


__all__ = [
    "HALLUCINATED_TOOL_PREFIX_RE",
    "recover_bare_call_tool_calls",
    "recover_hallucinated_tool_call",
    "recover_xml_hallucinated_tool_call",
    "recover_xml_hallucinated_tool_calls",
    "try_materialize_htc_tool_call",
    "validate_recovered_tool_args",
]
