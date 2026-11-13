import re

from langchain_core.tools import tool

from src.config import settings
from src.db.pool import get_pool
from src.kg.entities import lookup_kg_entity
from src.memory import AgentMemory, get_default_memory, sanitize_memory_fact
from src.memory_cleanup import (
    filter_deletable_memory_ids,
    forget_agent_memory,
    forget_memory_ids,
    memory_curation_status,
    parse_memory_ids,
    parse_uuid_or_none,
)
from src.memory_labels import world_fact_label_error

_UUID_IN_TEXT_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE
)


def _get_mem() -> AgentMemory:
    return get_default_memory()


def _gate_durable_memory_text(text: str) -> str | None:
    """Live: lexical provenance. Background: distill shape + provenance."""
    from src.reverie.distill import (
        gate_background_provenance,
        gate_solitary_memory_write,
        requires_distill_gate,
    )

    if requires_distill_gate():
        return gate_solitary_memory_write(text)
    return gate_background_provenance(text)


# Top-k default keeps ordinary recall cheap; sweeps ask for more explicitly.
_SEARCH_LIMIT_DEFAULT = 3
_SEARCH_LIMIT_MAX = 25


@tool
def search_memory(query: str, limit: int = _SEARCH_LIMIT_DEFAULT):
    """Search long-term memory. Use when the user asks about something discussed before.

    ``limit`` is how many rows to return (default 3, max 25). Raise it when
    sweeping rather than looking something up — "find ALL records about X",
    cleanup passes, or any batch you intend to feed to ``forget_memories``.
    Three hits cannot answer "all", and acting on a partial view silently
    leaves the rest of the job undone.
    """
    if not (query or "").strip():
        return []
    try:
        want = int(limit)
    except (TypeError, ValueError):
        want = _SEARCH_LIMIT_DEFAULT
    want = max(1, min(want, _SEARCH_LIMIT_MAX))
    results = _get_mem().search(query, limit=want, agent=settings.default_agent_id)
    return [f"[id={mid}] {content}" for mid, content, _score in results]


@tool
def memory_replace(memory_id: str, new_content: str):
    """Replace an existing memory by id. Use after search_memory returns an outdated fact."""
    mid = (memory_id or "").strip()
    uid = parse_uuid_or_none(mid)
    if not uid:
        return "not_found"
    text = (new_content or "").strip()
    if not text:
        return "empty_content"
    if len(text) > 4000:
        text = text[:4000]
    text = sanitize_memory_fact(text)
    if not text:
        return "injection_blocked"
    if world_fact_label_error(text) == "prefix_required":
        return "deixis_error: use USER: or AGENT: prefix, or diary for inner voice"
    try:
        rejected = _gate_durable_memory_text(text)
        if rejected:
            return rejected
    except Exception as exc:
        return f"distill_rejected: gate_failed ({type(exc).__name__})"
    pool = get_pool()
    with pool.connection() as conn:
        status = memory_curation_status(
            conn, agent_id=settings.default_agent_id, memory_id=uid
        )
        if status == "protected":
            return "protected"
        if status == "not_found":
            return "not_found"
    ok = _get_mem().replace(uid, text, agent=settings.default_agent_id)
    return "replaced" if ok else "not_found"


@tool
def forget_memory(memory_id: str):
    """Forget (archive and delete) a world-fact memory by id.

    Use when a retrieved fact is noise, outdated, or interferes with thinking.
    Identity and affirmation rows cannot be forgotten.
    """
    mid = (memory_id or "").strip()
    if not parse_uuid_or_none(mid):
        return "not_found"
    pool = get_pool()
    with pool.connection() as conn:
        status = forget_agent_memory(
            conn,
            agent_id=settings.default_agent_id,
            memory_id=mid,
            apply=True,
        )
    return status


@tool
def forget_memories(memory_ids: list[str] | str | dict):
    """Forget (archive and delete) MANY world-fact memories in one call.

    Use this for cleanup over a search result — pass every id to remove at
    once instead of chaining ``forget_memory`` per row.

    Preferred shape is a plain list of ids: ``["uuid-1", "uuid-2"]``. A
    comma-separated string or a mapping containing them also works — every uuid
    in the argument is used, wherever it sits. This tool only deletes, so any
    per-id value is ignored. Identity and affirmation rows are refused.

    Returns outcome counts: deleted / not_found / protected / invalid.
    """
    # Total by design: the caller is a model, so no input shape may lose the
    # ids. A rejected or misread call comes back as an opaque failure, the model
    # retries the same payload, and the turn burns rounds instead of deleting
    # (observed: `{"memory_ids": {"ids": [...]}}` — the uuid sat one level down,
    # reading dict keys yielded "ids", and the call reported no valid ids while
    # the reply claimed the row was gone).
    # Ids are uuids, so pattern-matching the whole payload finds them at any
    # nesting instead of guessing the wrapper the model picked this time.
    if isinstance(memory_ids, str):
        raw: list = parse_memory_ids(memory_ids)
        if not any(parse_uuid_or_none(str(x).strip()) for x in raw):
            raw = _UUID_IN_TEXT_RE.findall(memory_ids)
    else:
        raw = _UUID_IN_TEXT_RE.findall(str(memory_ids))
        if not raw and isinstance(memory_ids, list):
            raw = list(memory_ids)
    ids: list[str] = []
    invalid = 0
    for item in raw:
        canonical = parse_uuid_or_none(str(item).strip())
        if canonical:
            ids.append(canonical)
        else:
            invalid += 1
    if not ids:
        return f"no valid ids (invalid={invalid})"

    pool = get_pool()
    with pool.connection() as conn:
        allowed, blocked = filter_deletable_memory_ids(conn, ids)
        result = forget_memory_ids(conn, allowed, apply=True) if allowed else {}

    deleted = int(result.get("deleted", 0) or 0)
    missing = len(result.get("ids_missing", []) or [])
    parts = [f"deleted={deleted}"]
    if missing:
        parts.append(f"not_found={missing}")
    if blocked:
        parts.append(f"protected={len(blocked)}")
    if invalid:
        parts.append(f"invalid={invalid}")
    return " ".join(parts)


@tool
def save_memory(content: str):
    """Persist a labeled world fact (USER: … or AGENT: …, or neutral stack/world text).

    Inner voice / first-person "I" without prefix belongs in diary (reflect), not here.
    """
    content = (content or "").strip()
    if not content:
        return "empty_content"
    if len(content) > 4000:
        content = content[:4000]

    content = sanitize_memory_fact(content)
    if not content:
        return "injection_blocked"

    label_err = world_fact_label_error(content)
    if label_err == "prefix_required":
        return (
            "Nothing to save: first-person world fact needs USER: or AGENT: prefix; "
            "my inner voice belongs in diary (reflect), not save_memory."
        )

    try:
        rejected = _gate_durable_memory_text(content)
        if rejected:
            return rejected
    except Exception as exc:
        # Fail closed: must not dump when the provenance gate is broken.
        return f"distill_rejected: gate_failed ({type(exc).__name__})"

    mem = _get_mem()
    if bool(settings.reflect_consistency_gate_enabled):
        from src.graph.reflect_consistency import (
            apply_memory_contradiction,
            find_memory_contradiction,
        )

        conflict = find_memory_contradiction(mem, content)
        if conflict is not None:
            if not apply_memory_contradiction(mem, content, conflict):
                return "Nothing to save: contradiction gate blocked."
            return "saved (contradiction policy applied)"

    outcome = mem.save(
        content,
        memory_type="semantic",
        agent=settings.default_agent_id,
        source="tool",
    )
    if outcome is None:
        return "Nothing to save: skipped."
    mid, status = outcome
    if status == "dedup":
        return f"unchanged id={mid} (near-duplicate)"
    return f"saved id={mid}"


@tool
def find_related(name: str, depth: int = 2, predicates: list[str] | None = None):
    """Multi-hop walk from an entity by name. Use for structured relationships (owns, contains, etc.)."""
    raw = (name or "").strip()
    if not raw:
        return "No entity name provided."
    eid = lookup_kg_entity(raw, agent=settings.default_agent_id)
    if eid is None:
        return f"No entity found for: {raw}"
    d = max(1, min(int(depth), 3))
    norm_preds: list[str] | None = None
    if predicates:
        norm_preds = [
            p.strip().lower().replace(" ", "_")
            for p in predicates
            if isinstance(p, str) and p.strip()
        ] or None
    lim = max(1, int(settings.kg_find_related_limit))
    rows = _get_mem().traverse_graph(
        eid,
        depth=d,
        predicates=norm_preds,
        agent=settings.default_agent_id,
        max_edges=lim,
    )
    if not rows:
        return "No related edges found."
    return "\n".join(
        f"{r['subject']} · {r['predicate']} · {r['object']} (hop={r['hop']})"
        for r in rows[:lim]
    )


@tool("query_graph")
def query_graph_rows(
    subject: str | None = None,
    predicate: str | None = None,
    object_str: str | None = None,
) -> str:
    """Filter knowledge_graph triples by ILIKE on subject/predicate/object entity text."""
    lim = max(1, int(settings.kg_find_related_limit))
    rows = _get_mem().query_graph(
        subject=subject,
        predicate=predicate,
        object_=object_str,
        limit=lim,
        agent=settings.default_agent_id,
    )
    if not rows:
        return "No triples matched."
    return "\n".join(f"{s} · {p} · {o}" for s, p, o in rows)


@tool
def search_graph(query: str):
    """Search relevant facts in knowledge_graph by semantic similarity."""
    if not (query or "").strip():
        return []
    rows = _get_mem().search_graph(query, limit=5, agent=settings.default_agent_id)
    return [f"{s} - {p} - {o} (sim={sim:.2f})" for s, p, o, sim in rows]
