"""Individual read-only self-test checks."""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from urllib import request as urllib_request

import psycopg

from src.boot_checks import (
    EMBED_DIM,
    assert_database_ready,
    assert_ollama_ready,
    extract_embedding_vector,
)
from src.config import Settings
from src.ollama_client import ollama_http_client
from src.selftest.core import CheckResult, CheckStatus
from src.text_locale import ALLOWED_LANGUAGES

REQUIRED_TABLES = frozenset(
    {
        "memories",
        "entities",
        "knowledge_graph",
        "self_reflections",
        "conversations",
        "vitals",
        "agent_identity",
        "agent_activity",
        "outbox",
        "mind_events",
        "intents",
        "causal_lessons",
        "policy_knobs",
        "dpo_gold_pairs",
    }
)

CHECKPOINT_TABLES = frozenset(
    {
        "checkpoint_writes",
        "checkpoint_blobs",
        "checkpoint_migrations",
    }
)

EXPECTED_AGENT_TOOLS = frozenset(
    {
        "search_memory",
        "save_memory",
        "memory_replace",
        "forget_memories",
        "forget_memory",
        "search_diary",
        "diary_replace",
        "diary_save",
        "forget_diary",
        "search_graph",
        "query_graph",
        "find_related",
        "search_by_time",
        "search_web",
        "fetch_url",
        "browse_web",
        "schedule_intent",
        "list_intents",
        "cancel_intent",
        "run_selftest",
        "message_user",
        "record_lesson",
        "search_lessons",
    }
)

_KEEP_ALIVE_UNIT_RE = re.compile(
    r"^-?\d+(\.\d+)?(ns|us|µs|ms|s|m|h)$",
    re.IGNORECASE,
)


def _elapsed_ms(t0: float) -> int:
    return int((time.perf_counter() - t0) * 1000)


def _result(
    name: str,
    status: CheckStatus,
    detail: str,
    *,
    t0: float | None = None,
) -> CheckResult:
    return CheckResult(
        name=name,
        status=status,
        detail=detail,
        latency_ms=_elapsed_ms(t0) if t0 is not None else None,
    )


def _connect(settings: Settings):
    return psycopg.connect(settings.database_url_plain_postgresql, connect_timeout=5)


def _ollama_tags(settings: Settings) -> list[str]:
    url = settings.ollama_base_url.rstrip("/") + "/api/tags"
    with urllib_request.urlopen(url, timeout=5.0) as resp:
        if resp.status >= 400:
            raise RuntimeError(f"HTTP {resp.status}")
        payload = json.loads(resp.read().decode())
    models = payload.get("models") if isinstance(payload, dict) else []
    names: list[str] = []
    for item in models or []:
        if isinstance(item, dict):
            name = str(item.get("name") or "").strip()
            if name:
                names.append(name)
    return names


def _model_in_tags(model: str, tags: list[str]) -> bool:
    target = (model or "").strip()
    if not target:
        return False
    if target in tags:
        return True
    base = target.split(":")[0]
    return any(t == target or t.startswith(base + ":") or t == base for t in tags)


def check_postgres(settings: Settings, *, probe_network: bool = False) -> CheckResult:
    t0 = time.perf_counter()
    try:
        assert_database_ready()
        return _result("postgres", CheckStatus.OK, "SELECT 1 ok", t0=t0)
    except Exception as exc:
        return _result("postgres", CheckStatus.FAIL, str(exc), t0=t0)


def check_pgvector(settings: Settings, *, probe_network: bool = False) -> CheckResult:
    t0 = time.perf_counter()
    try:
        with _connect(settings) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM pg_extension WHERE extname = %s",
                    ("vector",),
                )
                row = cur.fetchone()
        if row:
            return _result(
                "pgvector", CheckStatus.OK, "extension vector present", t0=t0
            )
        return _result("pgvector", CheckStatus.FAIL, "extension vector missing", t0=t0)
    except Exception as exc:
        return _result("pgvector", CheckStatus.FAIL, str(exc), t0=t0)


def check_tables(settings: Settings, *, probe_network: bool = False) -> CheckResult:
    t0 = time.perf_counter()
    try:
        with _connect(settings) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT table_name FROM information_schema.tables
                    WHERE table_schema = 'public'
                    """)
                present = {str(r[0]) for r in cur.fetchall()}
        missing = sorted(REQUIRED_TABLES - present)
        if missing:
            return _result(
                "tables",
                CheckStatus.FAIL,
                f"missing: {', '.join(missing)}",
                t0=t0,
            )
        ckpt_missing = sorted(CHECKPOINT_TABLES - present)
        if ckpt_missing and (settings.checkpointer or "").strip().lower() == "postgres":
            return _result(
                "tables",
                CheckStatus.WARN,
                f"core ok; checkpoint tables missing: {', '.join(ckpt_missing)}",
                t0=t0,
            )
        return _result(
            "tables",
            CheckStatus.OK,
            f"{len(REQUIRED_TABLES)} core tables present",
            t0=t0,
        )
    except Exception as exc:
        return _result("tables", CheckStatus.FAIL, str(exc), t0=t0)


def check_ollama_reachable(
    settings: Settings, *, probe_network: bool = False
) -> CheckResult:
    t0 = time.perf_counter()
    try:
        assert_ollama_ready()
        tags = _ollama_tags(settings)
        return _result(
            "ollama_reachable",
            CheckStatus.OK,
            f"GET /api/tags ok ({len(tags)} models)",
            t0=t0,
        )
    except Exception as exc:
        return _result("ollama_reachable", CheckStatus.FAIL, str(exc), t0=t0)


def check_reasoning_model(
    settings: Settings, *, probe_network: bool = False
) -> CheckResult:
    t0 = time.perf_counter()
    model = settings.reasoning_model
    try:
        tags = _ollama_tags(settings)
        if _model_in_tags(model, tags):
            return _result(
                "reasoning_model",
                CheckStatus.OK,
                f"{model} in tags",
                t0=t0,
            )
        return _result(
            "reasoning_model",
            CheckStatus.FAIL,
            f"{model} not in ollama tags",
            t0=t0,
        )
    except Exception as exc:
        return _result("reasoning_model", CheckStatus.FAIL, str(exc), t0=t0)


def check_embed_model(
    settings: Settings, *, probe_network: bool = False
) -> CheckResult:
    t0 = time.perf_counter()
    model = settings.embed_model
    try:
        tags = _ollama_tags(settings)
        if not _model_in_tags(model, tags):
            return _result(
                "embed_model",
                CheckStatus.FAIL,
                f"{model} not in ollama tags",
                t0=t0,
            )
        resp = ollama_http_client().embeddings(
            model=model,
            prompt="ping",
            keep_alive=settings.llm_keep_alive,
        )
        embedding = extract_embedding_vector(resp)
        dim = len(embedding)
        if dim != EMBED_DIM:
            return _result(
                "embed_model",
                CheckStatus.FAIL,
                f"dim={dim}, expected {EMBED_DIM}",
                t0=t0,
            )
        return _result(
            "embed_model",
            CheckStatus.OK,
            f"{model} embed dim={dim}",
            t0=t0,
        )
    except Exception as exc:
        return _result("embed_model", CheckStatus.FAIL, str(exc), t0=t0)


def check_agent_tools(
    settings: Settings, *, probe_network: bool = False
) -> CheckResult:
    t0 = time.perf_counter()
    try:
        from src.graph import workflow

        bound = {
            str(getattr(t, "name", "") or "")
            for t in workflow.tools
            if getattr(t, "name", None)
        }
        missing = sorted(EXPECTED_AGENT_TOOLS - bound)
        extra = sorted(bound - EXPECTED_AGENT_TOOLS)
        if missing:
            return _result(
                "agent_tools",
                CheckStatus.FAIL,
                f"missing: {', '.join(missing)}",
                t0=t0,
            )
        if extra:
            return _result(
                "agent_tools",
                CheckStatus.WARN,
                f"extra tools: {', '.join(extra)}",
                t0=t0,
            )
        return _result(
            "agent_tools",
            CheckStatus.OK,
            f"{len(EXPECTED_AGENT_TOOLS)} tools bound",
            t0=t0,
        )
    except Exception as exc:
        return _result("agent_tools", CheckStatus.FAIL, str(exc), t0=t0)


def check_identity(settings: Settings, *, probe_network: bool = False) -> CheckResult:
    t0 = time.perf_counter()
    agent = settings.default_agent_id
    try:
        with _connect(settings) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT agent FROM agent_identity WHERE agent = %s LIMIT 1",
                    (agent,),
                )
                row = cur.fetchone()
        if row:
            return _result(
                "identity",
                CheckStatus.OK,
                f"agent_identity row for {agent!r}",
                t0=t0,
            )
        return _result(
            "identity",
            CheckStatus.WARN,
            f"no agent_identity row for {agent!r} (Ed25519 optional)",
            t0=t0,
        )
    except Exception as exc:
        return _result("identity", CheckStatus.WARN, str(exc), t0=t0)


_HEART_CMD_MARKERS = ("src.heart.heartbeat", "src/heart/heartbeat")


def _heart_pid_command(pid: int) -> str:
    try:
        out = subprocess.check_output(
            ["ps", "-p", str(pid), "-o", "command="],
            text=True,
            timeout=1.0,
        )
        return (out or "").strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _heart_pid_alive(pid: int) -> bool:
    """True only when *this* PID is a live Heart process — not a reused slot."""
    if not pid or pid < 1:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    except OSError:
        return False
    cmd = _heart_pid_command(pid).casefold()
    return any(marker in cmd for marker in _HEART_CMD_MARKERS)


def check_heartbeat(settings: Settings, *, probe_network: bool = False) -> CheckResult:
    t0 = time.perf_counter()
    mode = (settings.vitals_trust_mode or "strict").strip().lower()
    try:
        from src.heart.vitals import fetch_latest_vitals_sync

        row = fetch_latest_vitals_sync(connect_timeout=5.0)
        if row is None:
            detail = "heart silent — no vitals tick recorded"
            if mode == "dev":
                return _result("heartbeat", CheckStatus.WARN, detail, t0=t0)
            return _result("heartbeat", CheckStatus.FAIL, detail, t0=t0)
        tick_at = row.get("tick_at")
        if tick_at is None:
            detail = "heart silent — tick without timestamp"
            if mode == "dev":
                return _result("heartbeat", CheckStatus.WARN, detail, t0=t0)
            return _result("heartbeat", CheckStatus.FAIL, detail, t0=t0)
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        if tick_at.tzinfo is None:
            tick_at = tick_at.replace(tzinfo=timezone.utc)
        age_sec = (now - tick_at).total_seconds()
        max_age = 2 * max(1, int(settings.heart_interval_sec))

        heart_pid = row.get("heart_pid")
        pid_alive = _heart_pid_alive(heart_pid) if heart_pid else False
        sig = row.get("signature")
        signed = bool(sig and str(sig).strip())

        if age_sec <= max_age and signed:
            if pid_alive:
                detail = f"heart beating — {int(age_sec)}s ago, pid {heart_pid}"
            else:
                detail = f"heart beating — {int(age_sec)}s ago (pid {heart_pid} gone, likely restarted)"
            return _result("heartbeat", CheckStatus.OK, detail, t0=t0)
        if age_sec > max_age:
            if pid_alive:
                detail = f"heart slow — last beat {int(age_sec)}s ago but pid {heart_pid} alive"
            else:
                detail = (
                    f"heart stale — last beat {int(age_sec)}s ago, pid {heart_pid} gone"
                )
            if mode == "dev":
                return _result("heartbeat", CheckStatus.WARN, detail, t0=t0)
            return _result("heartbeat", CheckStatus.FAIL, detail, t0=t0)
        detail = "heart unsigned — tick present but not signed"
        if mode == "dev":
            return _result("heartbeat", CheckStatus.WARN, detail, t0=t0)
        return _result("heartbeat", CheckStatus.FAIL, detail, t0=t0)
    except Exception as exc:
        if mode == "dev":
            return _result("heartbeat", CheckStatus.WARN, str(exc), t0=t0)
        return _result("heartbeat", CheckStatus.FAIL, str(exc), t0=t0)


check_vitals_freshness = check_heartbeat


def check_memory_stats(
    settings: Settings, *, probe_network: bool = False
) -> CheckResult:
    t0 = time.perf_counter()
    try:
        with _connect(settings) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM memories")
                mem_n = int(cur.fetchone()[0])
                cur.execute("SELECT COUNT(*) FROM knowledge_graph")
                kg_n = int(cur.fetchone()[0])
                cur.execute("SELECT COUNT(*) FROM self_reflections")
                refl_n = int(cur.fetchone()[0])
        return _result(
            "memory_stats",
            CheckStatus.OK,
            f"memories={mem_n} triples={kg_n} reflections={refl_n}",
            t0=t0,
        )
    except Exception as exc:
        return _result(
            "memory_stats",
            CheckStatus.FAIL,
            f"counts unavailable: {exc}",
            t0=t0,
        )


def _web_fetch_probe_failed(body: str) -> bool:
    """True when ``fetch_url_text`` output indicates failure, not a live page."""
    low = (body or "").strip().lower()
    if not low:
        return True
    if low.startswith("fetch failed") or low.startswith("url rejected"):
        return True
    if low.startswith("http "):
        return True
    if "web fetch is disabled" in low:
        return True
    if "too large" in low and "bytes" in low:
        return True
    if "(no readable text extracted)" in low:
        return True
    return False


def check_config_sanity(
    settings: Settings, *, probe_network: bool = False
) -> CheckResult:
    t0 = time.perf_counter()
    warnings: list[str] = []
    for label, val in (("LLM_KEEP_ALIVE", settings.llm_keep_alive),):
        raw = str(val or "").strip()
        if not raw:
            warnings.append(f"{label} empty")
        elif not _KEEP_ALIVE_UNIT_RE.match(raw):
            warnings.append(f"{label}={raw!r} missing unit")
    if not ALLOWED_LANGUAGES:
        warnings.append("REPLY_LANGUAGES empty (ALLOWED_LANGUAGES)")
    if warnings:
        return _result(
            "config_sanity",
            CheckStatus.WARN,
            "; ".join(warnings),
            t0=t0,
        )
    langs = ",".join(sorted(ALLOWED_LANGUAGES))
    return _result(
        "config_sanity",
        CheckStatus.OK,
        f"keep_alive ok; REPLY_LANGUAGES={langs}",
        t0=t0,
    )


def check_web_tools(settings: Settings, *, probe_network: bool = False) -> CheckResult:
    t0 = time.perf_counter()
    if not probe_network:
        return _result(
            "web_tools",
            CheckStatus.SKIP,
            "live probe disabled (use --probe-network)",
            t0=t0,
        )
    if not settings.web_fetch_enabled and not settings.web_search_enabled:
        return _result(
            "web_tools",
            CheckStatus.WARN,
            "WEB_FETCH_ENABLED and WEB_SEARCH_ENABLED both off",
            t0=t0,
        )
    if not settings.web_fetch_enabled:
        return _result(
            "web_tools",
            CheckStatus.WARN,
            "WEB_FETCH_ENABLED=0; fetch probe skipped",
            t0=t0,
        )
    try:
        from src.tools.web_core import fetch_url_text

        body = fetch_url_text("https://example.com/")
        if _web_fetch_probe_failed(body):
            return _result("web_tools", CheckStatus.FAIL, body[:120], t0=t0)
        return _result(
            "web_tools",
            CheckStatus.OK,
            f"fetch example.com ok ({len(body)} chars)",
            t0=t0,
        )
    except Exception as exc:
        return _result("web_tools", CheckStatus.FAIL, str(exc), t0=t0)


def check_role_review_schema(
    settings: Settings, *, probe_network: bool = False
) -> CheckResult:
    """ROLE_REVIEW=1 requires ``dpo_gold_pairs`` (migration)."""
    t0 = time.perf_counter()
    if not bool(settings.role_review):
        return _result(
            "role_review_schema",
            CheckStatus.SKIP,
            "role_review disabled",
            t0=t0,
        )
    try:
        with _connect(settings) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT to_regclass('public.dpo_gold_pairs')")
                row = cur.fetchone()
        if row is None or row[0] is None:
            return _result(
                "role_review_schema",
                CheckStatus.FAIL,
                "role_review enabled but dpo_gold_pairs missing — run: make migrate",
                t0=t0,
            )
        return _result(
            "role_review_schema",
            CheckStatus.OK,
            "dpo_gold_pairs present",
            t0=t0,
        )
    except Exception as exc:
        return _result("role_review_schema", CheckStatus.FAIL, str(exc), t0=t0)
