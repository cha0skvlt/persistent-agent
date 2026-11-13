from __future__ import annotations

import logging
import re
from urllib.parse import urlparse

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_log = logging.getLogger(__name__)


def _normalize_pg_url_for_parse(url: str) -> str:
    """Strip SQLAlchemy driver prefix so ``urlparse`` sees host/port/path."""
    if url.startswith("postgresql+psycopg://"):
        return "postgresql://" + url.removeprefix("postgresql+psycopg://")
    return url


class Settings(BaseSettings):
    """Application configuration (env + ``.env``). One source of truth for defaults."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    """Agent Core release version (API/CLI status)."""

    # --- DB_* (PostgreSQL + pool) ---
    database_url: str = "postgresql://localhost:5432/agent_db"
    """``DATABASE_URL``: psycopg/async URL used by the app and Alembic (after driver rewrite in env.py)."""
    db_pool_min: int = 1
    """``DB_POOL_MIN``: minimum connections in the app (memory/KG) pool."""
    db_pool_timeout: float = 30.0
    """``DB_POOL_TIMEOUT``: seconds to wait for a pooled connection before raising."""
    db_pool_max: int = 10
    """``DB_POOL_MAX``: maximum connections in the app pool."""
    db_checkpoint_pool_min: int = 1
    """``DB_CHECKPOINT_POOL_MIN``: minimum connections in the LangGraph checkpointer pool."""
    db_checkpoint_pool_max: int = 3
    """``DB_CHECKPOINT_POOL_MAX``: maximum connections reserved for PostgresSaver (isolated from app pool)."""

    # --- LLM_* (Ollama chat sampling / runtime) ---
    llm_keep_alive: str = "30m"
    """``LLM_KEEP_ALIVE``: Ollama ``keep_alive`` for the chat model (and embeddings unless overridden)."""
    llm_num_ctx: int = 32768
    """``LLM_NUM_CTX``: context window tokens for the chat model (prompt + generation)."""
    llm_num_predict: int = 8192
    """``LLM_NUM_PREDICT``: max tokens to generate per completion (thinking + visible reply share this budget)."""
    llm_reasoning: bool = True
    """``LLM_REASONING``: enable Ollama native thinking (``reasoning=True``). Disable for models without thinking support (e.g. llama3.3)."""
    llm_supports_tools: bool = True
    """``LLM_SUPPORTS_TOOLS``: model supports Ollama native tool calling. Disable for models without tool support (e.g. dolphin-llama3); agent falls back to HTC recovery for tool intent."""

    # --- Model (single chat LLM) ---
    reasoning_model: str = "qwen3:8b"
    """``REASONING_MODEL``: chat model for agent, summary, reflection."""

    # --- Ollama service (HTTP) ---
    ollama_base_url: str = "http://127.0.0.1:11434"
    """``OLLAMA_BASE_URL``: Ollama HTTP API root (no trailing slash). Used by the chat client and embedding client."""

    # --- Provider selection (chat + embeddings) ---
    llm_provider: str = "ollama"
    """``LLM_PROVIDER``: ``ollama`` | ``openai`` (vLLM, OpenAI, any OpenAI-compatible server)."""
    llm_base_url: str = ""
    """``LLM_BASE_URL``: chat endpoint. Empty = ``OLLAMA_BASE_URL`` for ollama; required for openai."""
    llm_api_key: str = ""
    """``LLM_API_KEY``: bearer token for an OpenAI-compatible chat endpoint."""
    llm_extra_body: str = ""
    """``LLM_EXTRA_BODY``: JSON object merged into the request body for OpenAI-compatible
    servers (e.g. ``{\"top_k\": 40, \"min_p\": 0.0}`` for vLLM). Empty = none."""
    embed_provider: str = "ollama"
    """``EMBED_PROVIDER``: ``ollama`` | ``openai``. Independent of ``LLM_PROVIDER``."""
    embed_base_url: str = ""
    """``EMBED_BASE_URL``: embedding endpoint. Empty = ``LLM_BASE_URL`` then ``OLLAMA_BASE_URL``."""
    embed_api_key: str = ""
    """``EMBED_API_KEY``: bearer token for an OpenAI-compatible embedding endpoint."""
    embed_dim: int = 768
    """``EMBED_DIM``: embedding dimension. Must match the ``vector(N)`` columns."""

    # --- EMBED_* (vector API) ---
    embed_model: str = "nomic-embed-text-v2-moe"
    """``EMBED_MODEL``: embedding model name (768-d multilingual by default)."""

    # --- Agent / LangGraph ---
    checkpointer: str = "postgres"
    """``CHECKPOINTER``: ``postgres`` | ``memory``."""
    checkpoint_auto_repair: bool = False
    """``CHECKPOINT_AUTO_REPAIR``: allow destructive checkpoint schema repair when legacy BYTEA is detected."""
    checkpoint_auto_repair_confirm: bool = False
    """``CHECKPOINT_AUTO_REPAIR_CONFIRM``: must be set with ``CHECKPOINT_AUTO_REPAIR`` to DROP legacy checkpoint tables."""
    default_agent_id: str = "agent"
    """``DEFAULT_AGENT_ID``: agent callsign for DB rows (``agent``). Separate from prompt-level persona text."""
    langgraph_thread_id: str = "main_session_v1"
    """``LANGGRAPH_THREAD_ID``: PostgresSaver thread id for REPL session continuity."""
    recursion_limit: int = 100
    """``RECURSION_LIMIT``: LangGraph max supersteps per invoke (graph nodes + tool loops).

    Backstop for the backstop. The working brake is ``graph/loop_budget.py``
    (stalling + wall clock); ``MAX_TOOL_ROUNDS_PER_TURN`` bounds a runaway. Each
    tool round costs 2 supersteps (agent + tools) on top of ~5 fixed nodes, so
    the 30-round ceiling plus web bonuses lands near 71 — this must stay above
    that or it becomes the hard cut it exists to avoid."""
    max_tool_rounds_per_turn: int = 30
    """``MAX_TOOL_ROUNDS_PER_TURN``: hard backstop on ToolNode runs per turn (0 = unlimited).

    Not the working brake. A long chain (search → read → act → verify) is normal
    work and must not be cut by a counter; ``src/graph/loop_budget.py`` stops the
    turn on *stalling* instead. This number only bounds a runaway."""
    loop_wall_clock_seconds: float = 240.0
    """``LOOP_WALL_CLOCK_SECONDS``: max wall-clock for one live turn's tool loop (0 = off)."""
    loop_stall_repeat_threshold: int = 2
    """``LOOP_STALL_REPEAT_THRESHOLD``: identical tool batch seen this many times in a turn = stalling."""
    loop_stall_fruitless_rounds: int = 3
    """``LOOP_STALL_FRUITLESS_ROUNDS``: consecutive rounds returning only errors/empties = stalling."""
    reverie_max_tool_rounds: int = 15
    """``REVERIE_MAX_TOOL_ROUNDS``: tool-round base cap for background passes.

    When > 0 this *is* the background cap (plus curiosity/essay extras), not
    ``max(live, this)``. ``0`` falls through to ``MAX_TOOL_ROUNDS_PER_TURN``."""
    checkpoint_max_messages: int = 48
    """``CHECKPOINT_MAX_MESSAGES``: max messages retained in the LangGraph checkpoint
    after a turn (trim). Keep in lockstep with ``CONTEXT_WINDOW_MESSAGES``."""
    tool_collapse_keep_recent_pairs: int = 12
    """``TOOL_COLLAPSE_KEEP_RECENT_PAIRS``: full tool call+result pairs kept at tail (older → archived)."""
    conversation_snapshot_max_messages: int = 4000
    """``CONVERSATION_SNAPSHOT_MAX_MESSAGES``: max JSON objects kept in ``conversations.messages`` when merging."""
    context_window_messages: int = 48
    """``CONTEXT_WINDOW_MESSAGES``: chat-history tail when building the LLM prompt.
    Must match ``CHECKPOINT_MAX_MESSAGES`` so the window is not an illusion."""
    token_est_cpt_latin: float = 3.4
    """``TOKEN_EST_CPT_LATIN``: chars-per-token for non-Cyrillic text in prompt budget estimates (lower = more conservative)."""
    token_est_cpt_cyrillic: float = 2.0
    """``TOKEN_EST_CPT_CYRILLIC``: chars-per-token for Cyrillic (Qwen denser than Latin; lower = more conservative)."""
    sys_mem_max_chars: int = 4000
    """``SYS_MEM_MAX_CHARS``: max chars for the [MEM] block in agent SYS (tail-truncated)."""
    sys_summary_max_chars: int = 2000
    """``SYS_SUMMARY_MAX_CHARS``: max chars for session summary in agent SYS (tail-truncated)."""
    sys_mind_max_chars: int = 2400
    """``SYS_MIND_MAX_CHARS``: max chars for Mind Hub SYS block (tail-truncated; 0 = uncapped)."""
    sys_diary_max_chars: int = 2000
    """``SYS_DIARY_MAX_CHARS``: max chars for diary/self-reflection SYS block (tail-truncated; 0 = uncapped)."""
    llm_max_concurrent_turns: int = 1
    """``LLM_MAX_CONCURRENT_TURNS``: max parallel live graph invokes (32B GPU); background jobs use the same model tag."""

    # --- MEMORY_* (semantic memory) ---
    memory_dedup_cosine: float = Field(
        default=0.92,
        validation_alias=AliasChoices("MEMORY_DEDUP_COSINE", "MEMORY_DEDUP_THRESHOLD"),
    )
    """``MEMORY_DEDUP_COSINE``: cosine ≥ this → duplicate (do not set below 0.88)."""
    memory_dedup_neighbors: int = 5
    """``MEMORY_DEDUP_NEIGHBORS``: nearest vectors checked on save (top-k dedup)."""
    memory_dedup_fuzzy_cosine: float = 0.85
    """``MEMORY_DEDUP_FUZZY_COSINE``: cosine ≥ this + text overlap → duplicate."""
    memory_dedup_text_overlap: float = 0.8
    """``MEMORY_DEDUP_TEXT_OVERLAP``: min |new∩existing|/|new| for fuzzy dedup."""
    memory_dedup_base_importance: float = 0.5
    """``MEMORY_DEDUP_BASE_IMPORTANCE``: assumed importance when column is null during dedup."""
    memory_importance_cap: float = 1.0
    """``MEMORY_IMPORTANCE_CAP``: maximum stored importance."""
    memory_search_ann_candidates: int = 50
    """``MEMORY_SEARCH_ANN_CANDIDATES``: HNSW shortlist size before rerank (0 = legacy full scan)."""
    memory_ttl_days: int = 365
    """``MEMORY_TTL_DAYS``: retention horizon for forget tooling."""
    memory_forget_min_importance: float = 0.1
    """``MEMORY_FORGET_MIN_IMPORTANCE``: forget script threshold."""

    # --- SUMMARY_* ---
    summary_every_n: int = 20
    """``SUMMARY_EVERY_N``: run rolling summary every N turns (when enabled)."""
    summary_max_chars: int = 1500
    """``SUMMARY_MAX_CHARS``: max length of stored session summary text."""
    summary_llm_max_messages: int = 80
    """``SUMMARY_LLM_MAX_MESSAGES``: max messages sent to the summarizer LLM."""
    summary_llm_max_messages_force: int = 24
    """``SUMMARY_LLM_MAX_MESSAGES_FORCE``: cap for explicit forced summaries."""
    summary_defer: bool = True
    """``SUMMARY_DEFER``: run session summary after the user-visible reply (not before agent)."""

    # --- REFLECT_* ---
    reflection: bool = True
    """``REFLECTION``: enable reflect_node."""
    reflect_every_n: int = 5
    """``REFLECT_EVERY_N``: run reflection when ``turn_count`` is a multiple of N."""
    reflect_skip_short_reply_chars: int = 450
    """``REFLECT_SKIP_SHORT_REPLY_CHARS``: skip reflect when agent reply is shorter (no tools this turn)."""
    reflect_defer: bool = True
    """``REFLECT_DEFER``: run reflect_node after the user-visible reply (not before)."""
    self_reflect_every_n: int = 10
    """``SELF_REFLECT_EVERY_N``: run first-person self-reflection when ``turn_count`` % N == 0."""

    # --- SOLITARY_* (reverie cycle) ---
    solitary_enabled: bool = False
    """``SOLITARY_ENABLED``: enable background solitary reflection cycle in API process."""
    solitary_interval_min: int = 5
    """``SOLITARY_INTERVAL_MIN``: minimum minutes between consecutive solitary passes (0 = no gap)."""
    reverie_loop_min_sec: int = 30
    """``REVERIE_LOOP_MIN_SEC``: seconds between background loop ticks when interval_min is 0."""

    # --- OUTREACH_* (proactive outreach) ---
    outreach_enabled: bool = False
    """``OUTREACH_ENABLED``: global kill switch for proactive Telegram messages.

    No cooldown, no dedup, no idle gate — whether and how often to write is the
    agent's own decision at queue time (``message_user``)."""
    outreach_queue_max: int = 8
    """``OUTREACH_QUEUE_MAX``: max pending outbox rows; ``message_user`` refuses above this (anti-spam)."""
    outreach_footer_grace_sec: float = 90.0
    """``OUTREACH_FOOTER_GRACE_SEC``: Heart skips pending outbox rows with empty ``footer_text`` younger than this (wait for post-graph stamp). After grace, send with a synthetic ``→ turn`` line."""
    novelty_cosine_threshold: float = 0.88
    """``NOVELTY_COSINE_THRESHOLD``: embedding cosine above which a **diary** entry is treated as duplicate (memory hygiene only — never gates outreach)."""

    # --- RHYTHM_* (alias over solitary interval) ---
    rhythm_solitary_interval_min: int | None = None
    """``RHYTHM_SOLITARY_INTERVAL_MIN``: alias for SOLITARY_INTERVAL_MIN (None = use solitary_*)."""

    # --- MIND_HUB_* (cross-channel operational timeline) ---
    mind_hub_enabled: bool = True
    """``MIND_HUB_ENABLED``: inject unified live/solitary/outreach timeline into each turn."""
    mind_hub_max_events: int = 12
    """``MIND_HUB_MAX_EVENTS``: max recent mind_events lines injected per turn."""

    # --- PULSE_* (scheduled intentions) ---
    pulse_enabled: bool = True
    """``PULSE_ENABLED``: global kill switch for intent scheduling and execution."""
    intents_max_pending: int = 20
    """``INTENTS_MAX_PENDING``: max pending/due intents per agent (schedule_intent cap)."""
    intents_max_per_day: int = 10
    """``INTENTS_MAX_PER_DAY``: max open intents (pending/due/running) created per agent in a rolling 24h window; cancelled/done do not count."""
    intents_running_ttl_min: int = 30
    """``INTENTS_RUNNING_TTL_MIN``: minutes before a stuck running intent is recovered to due."""
    pulse_sys_max_chars: int = 600
    """``PULSE_SYS_MAX_CHARS``: max chars for time-sense SYS block (tail-truncate)."""

    # --- ROLE_REVIEW_* (offline DPO gold pairs) ---
    role_review: bool = False
    """``ROLE_REVIEW``: after reply, critique last turn for agent behavior; save gold pairs when review failed."""
    role_review_min_reply_chars: int = 40
    """``ROLE_REVIEW_MIN_REPLY_CHARS``: skip review when agent reply is shorter."""
    role_review_every_n: int = 3
    """``ROLE_REVIEW_EVERY_N``: run role review when ``turn_count`` % N == 0 (reduces GPU queue load)."""

    # --- KG_* ---
    kg_entity_resolve_threshold: float = 0.90
    """``KG_ENTITY_RESOLVE_THRESHOLD``: cosine merge threshold for entity resolution."""
    kg_find_related_limit: int = 30
    """``KG_FIND_RELATED_LIMIT``: max rows returned for graph walks / query display."""

    vitals_trust_mode: str = "dev"
    """``VITALS_TRUST_MODE``: ``strict`` (signed fresh heart required) | ``dev`` (no emergency when vitals missing/unsigned)."""

    @field_validator("vitals_trust_mode", mode="before")
    @classmethod
    def _normalize_vitals_trust_mode(cls, value: object) -> str:
        raw = str(value or "dev").strip().lower()
        if raw in {"strict", "dev"}:
            return raw
        _log.warning("Invalid VITALS_TRUST_MODE=%r; using strict", value)
        return "strict"

    # --- retrieve_memory tuning ---
    memory_retrieve_top_k: int = 3
    """``MEMORY_RETRIEVE_TOP_K``: max [MEM] lines injected after rerank."""
    memory_similarity_floor: float = 0.3
    """``MEMORY_SIMILARITY_FLOOR``: drop candidates below this query similarity."""
    memory_distrust_enabled: bool = True
    """``MEMORY_DISTRUST_ENABLED``: apply retrieve-side downrank for unverified/aged memories."""
    memory_retrieve_types: str = "semantic,unverified"
    """``MEMORY_RETRIEVE_TYPES``: comma-separated ``memory_type`` values eligible for [MEM] search."""
    memory_unverified_rerank_factor: float = 0.35
    """``MEMORY_UNVERIFIED_RERANK_FACTOR``: score multiplier for ``memory_type=unverified`` (``(0,1]``)."""
    memory_age_half_life_days: float = 180.0
    """``MEMORY_AGE_HALF_LIFE_DAYS``: age decay half-life in days (``0`` disables). Identity rows skip age decay."""
    memory_contradiction_importance_factor: float = 0.7
    """``MEMORY_CONTRADICTION_IMPORTANCE_FACTOR``: multiply existing row importance on contradiction for ``log`` / ``unverified`` (``replace`` returns early)."""
    lesson_retrieve_top_k: int = 2
    """``LESSON_RETRIEVE_TOP_K``: max causal lesson lines injected after retrieve (``0`` disables)."""
    self_reflection_retrieve_top_k: int = 4
    """``SELF_REFLECTION_RETRIEVE_TOP_K``: max diary lines injected after semantic search."""
    diary_echo_check_n: int = 20
    """``DIARY_ECHO_CHECK_N``: recent diary rows to compare in deferred reflect echo gate."""
    diary_echo_threshold: float = 0.92
    """``DIARY_ECHO_THRESHOLD``: max cosine vs recent diary above which deferred reflect skips save."""

    reflect_consistency_gate_enabled: bool = False
    """``REFLECT_CONSISTENCY_GATE_ENABLED``: consistency check before world-fact writes (``save_memory`` tool + deferred reflect) and deferred diary writes. Off by default: the negation-polarity heuristic is too weak an oracle — echo gate + distill keep write hygiene."""
    reflect_consistency_gate_top_k: int = 5
    """``REFLECT_CONSISTENCY_GATE_TOP_K``: similar-row candidates to fetch for consistency gate (same embed+cosine path as search_memory/search_diary)."""
    memory_contradiction_policy: str = "log"
    """``MEMORY_CONTRADICTION_POLICY``: ``log`` | ``unverified`` | ``replace`` (``save_memory`` + deferred reflect)."""
    reflect_consistency_gate_sim_threshold: float | None = None
    """``REFLECT_CONSISTENCY_GATE_SIM_THRESHOLD``: optional override (0..1). When unset, derived from ``MEMORY_SIMILARITY_FLOOR``."""

    @field_validator("reflect_consistency_gate_sim_threshold", mode="before")
    @classmethod
    def _empty_reflect_sim_threshold(cls, value: object) -> object:
        if value is None or value == "":
            return None
        return value

    conversation_snapshot_every_n: int = 3
    """``CONVERSATION_SNAPSHOT_EVERY_N``: REPL persists ``conversations`` every N turns (1 = each turn)."""

    # --- Phylactery identity ---
    sign_responses: bool = False
    """``SIGN_RESPONSES``: append persona/sig footer to agent responses."""

    # --- Heart daemon ---
    heart_interval_sec: int = 30
    """``HEART_INTERVAL_SEC``: seconds between vitals ticks."""
    heart_backup_interval_sec: int = 21_600
    """``HEART_BACKUP_INTERVAL_SEC``: heart runs encrypted pg_dump on this interval (0 = off)."""
    heart_notify_chat_id: str = ""
    """``HEART_NOTIFY_CHAT_ID``: Telegram chat id for heart alerts (empty = notify disabled)."""
    heart_notify_cooldown_sec: int = 600
    """``HEART_NOTIFY_COOLDOWN_SEC``: min seconds between alerts of the same type."""
    heart_disk_alert_pct: float = 95.0
    """``HEART_DISK_ALERT_PCT``: disk used %% at or above → critical notify (≈5%% free)."""
    heart_battery_alert_pct: float = 5.0
    """``HEART_BATTERY_ALERT_PCT``: battery below this → critical notify."""
    heart_ollama_down_alert_sec: int = 300
    """``HEART_OLLAMA_DOWN_ALERT_SEC``: Ollama unreachable this long → critical notify."""
    heart_check_api_enabled: bool = True
    """``HEART_CHECK_API_ENABLED``: poll API ``/health`` each heart tick."""
    heart_api_health_url: str = "http://127.0.0.1:8080/health"
    """``HEART_API_HEALTH_URL``: health endpoint for API watchdog."""
    heart_api_fail_threshold: int = 3
    """``HEART_API_FAIL_THRESHOLD``: consecutive fails before ``emergency_api_dead``."""
    heart_api_auto_restart: bool = False
    """``HEART_API_AUTO_RESTART``: ``launchctl kickstart -k`` when API emergency triggers."""
    heart_api_launchd_label: str = ""
    """``HEART_API_LAUNCHD_LABEL``: launchd target for API restart (e.g. ``gui/501/cha0skvlt.persistent-agent.api``)."""
    phylactery_backup_dir: str = "~/Backups/agent"
    """``PHYLACTERY_BACKUP_DIR``: encrypted custom-format dumps."""
    phylactery_max_backups: int = 28
    """``PHYLACTERY_MAX_BACKUPS``: rolling retention for *.dump.enc."""
    phylactery_restore_db: str = "agent_db_restore"
    """``PHYLACTERY_RESTORE_DB``: pg_restore target DB name — never overwrites primary DB."""

    # --- Boot ---
    boot_retry_attempts: int = 30
    """``BOOT_RETRY_ATTEMPTS``: how many times API lifespan retries ``verify_boot_dependencies`` before giving up."""
    boot_retry_sleep_sec: float = 10.0
    """``BOOT_RETRY_SLEEP_SEC``: seconds between boot dependency retries."""

    # --- Time / OS services ---
    agent_tz: str = "UTC"
    """``AGENT_TZ``: IANA timezone for ``update_time`` and date prefixes."""
    postgres_service: str = "postgresql@18"
    """``POSTGRES_SERVICE``: Homebrew services label for ``brew services start`` (start.sh)."""

    # --- Logging ---
    log_rotate_max_mb: int = 64
    """``LOG_ROTATE_MAX_MB``: copytruncate ``agent-*.log`` above this size (heart tick)."""
    log_rotate_backups: int = 2
    """``LOG_ROTATE_BACKUPS``: rotated ``.1``/``.2`` copies kept per log file."""
    log_level: str = "INFO"
    """``LOG_LEVEL``: ``OFF`` (errors only, no BOOT) | ``INFO`` (default) | ``DEBUG`` (full technical trace)."""

    # --- OpenTelemetry (optional OTLP export; off by default) ---
    otel_enabled: bool = Field(default=False, validation_alias="OTEL_ENABLED")
    """``OTEL_ENABLED``: optional OTLP export (off by default; no in-repo collector stack)."""
    otel_endpoint: str = Field(
        default="127.0.0.1:4318",
        validation_alias="OTEL_EXPORTER_OTLP_ENDPOINT",
    )
    """``OTEL_EXPORTER_OTLP_ENDPOINT``: OTLP/HTTP endpoint (host:port or URL; 4317→4318)."""
    environment: str = Field(default="dev", validation_alias="ENVIRONMENT")
    """``ENVIRONMENT``: resource attribute ``deployment.environment``."""

    # --- Tests ---
    test_database_url: str | None = None
    """``TEST_DATABASE_URL``: disposable Postgres for destructive pytest (optional)."""

    # --- API / Telegram ---
    api_host: str = "127.0.0.1"
    """``API_HOST``: bind address for FastAPI / uvicorn."""
    api_port: int = 8080
    """``API_PORT``: bind port for FastAPI / uvicorn."""
    api_key: str = ""
    """``API_KEY``: optional Bearer token for ``POST /chat`` (empty = no auth)."""
    api_default_thread_id: str = "api_session_v1"
    """``API_DEFAULT_THREAD_ID``: LangGraph checkpoint thread for REST ``/chat``."""
    telegram_bot_token: str = ""
    """``TELEGRAM_BOT_TOKEN``: BotFather token; enables Telegram polling when set."""
    telegram_allowed_chat_ids: list[int] | str = Field(default_factory=list)
    """``TELEGRAM_ALLOWED_CHAT_IDS``: comma-separated chat ids allowed to talk to the bot.

    **Breaking:** empty list denies all chats (fail-closed). Set your numeric chat id
    before deploy, or set ``TELEGRAM_OWNER_CHAT_ID`` as a fallback allowlist entry.
    """

    telegram_owner_chat_id: int | str = ""
    """``TELEGRAM_OWNER_CHAT_ID``: optional fallback when allowlist is empty (owner only)."""

    @field_validator("telegram_owner_chat_id", mode="before")
    @classmethod
    def _parse_telegram_owner_chat_id(cls, value: object) -> int | str:
        if value is None or value == "":
            return ""
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            s = value.strip()
            if not s:
                return ""
            try:
                return int(s)
            except ValueError:
                _log.warning("Ignoring invalid TELEGRAM_OWNER_CHAT_ID: %r", value)
                return ""
        return ""

    @field_validator("telegram_allowed_chat_ids", mode="before")
    @classmethod
    def _parse_telegram_allowed_chat_ids(cls, value: object) -> list[int]:
        if value is None or value == "":
            return []
        out: list[int] = []
        if isinstance(value, list):
            for x in value:
                try:
                    out.append(int(x))
                except (TypeError, ValueError):
                    _log.warning(
                        "Ignoring invalid TELEGRAM_ALLOWED_CHAT_IDS entry: %r", x
                    )
            return out
        if isinstance(value, int):
            return [value]
        if isinstance(value, str):
            for part in value.split(","):
                part = part.strip()
                if not part:
                    continue
                try:
                    out.append(int(part))
                except ValueError:
                    _log.warning(
                        "Ignoring invalid TELEGRAM_ALLOWED_CHAT_IDS entry: %r", part
                    )
            return out
        return []

    telegram_polling: bool = True
    """``TELEGRAM_POLLING``: run getUpdates loop inside the API process."""
    telegram_poll_timeout: int = 30
    """``TELEGRAM_POLL_TIMEOUT``: long-poll seconds for Telegram getUpdates."""
    telegram_thread_prefix: str = "telegram"
    """``TELEGRAM_THREAD_PREFIX``: checkpoint thread id prefix per chat."""
    telegram_typing_interval_sec: float = 4.0
    """``TELEGRAM_TYPING_INTERVAL_SEC``: refresh ``sendChatAction(typing)`` interval (max ~5s UI)."""
    agent_mode_path: str = ""
    """``AGENT_MODE_PATH``: sleep/wake state file shared by API + Heart (empty = ``~/.cache/persistent-agent/agent_mode.json``)."""

    # --- CLI ---
    cli_default_thread_id: str = "cli_session_v1"
    """``CLI_DEFAULT_THREAD_ID``: LangGraph checkpoint thread for ``agent chat``."""

    # --- Web tools ---
    web_fetch_enabled: bool = True
    """``WEB_FETCH_ENABLED``: allow ``fetch_url`` tool."""
    web_search_enabled: bool = True
    """``WEB_SEARCH_ENABLED``: allow ``search_web`` tool."""
    web_fetch_timeout_sec: float = 25.0
    """``WEB_FETCH_TIMEOUT_SEC``: HTTP timeout for ``fetch_url``."""
    web_fetch_max_bytes: int = 1_500_000
    """``WEB_FETCH_MAX_BYTES``: max downloaded bytes per URL."""
    web_fetch_max_chars: int = 28_000
    """``WEB_FETCH_MAX_CHARS``: max plain text returned to the LLM."""
    web_evidence_max_chars: int = 4_500
    """``WEB_EVIDENCE_MAX_CHARS``: max prefetch body injected into the agent system prompt."""
    web_fetch_render_fallback: bool = True
    """``WEB_FETCH_RENDER_FALLBACK``: escalate ``fetch_url`` to a one-shot local Playwright render when static fetch is blocked or yields thin text."""
    web_fetch_render_min_chars: int = 200
    """``WEB_FETCH_RENDER_MIN_CHARS``: static page text shorter than this triggers the JS render fallback."""
    web_search_max_results: int = 5
    """``WEB_SEARCH_MAX_RESULTS``: results per ``search_web`` call."""
    web_search_max_retries: int = 3
    """``WEB_SEARCH_MAX_RETRIES``: search attempts on transient errors."""
    web_search_total_budget_sec: float = 20.0
    """``WEB_SEARCH_TOTAL_BUDGET_SEC``: wall-clock cap for one backend-chain pass (turn holds the LLM semaphore)."""
    web_backend_fail_threshold: int = 3
    """``WEB_BACKEND_FAIL_THRESHOLD``: consecutive failures that open a search backend's circuit breaker."""
    web_backend_cooldown_sec: float = 120.0
    """``WEB_BACKEND_COOLDOWN_SEC``: seconds an open breaker skips its backend before retrying."""
    web_search_query_max_chars: int = 160
    """``WEB_SEARCH_QUERY_MAX_CHARS``: max chars sent to search backends (focused query)."""
    web_search_region: str = "wt-wt"
    """``WEB_SEARCH_REGION``: DuckDuckGo region code (e.g. wt-wt, ru-ru)."""
    web_search_user_agent: str = ""
    """``WEB_SEARCH_USER_AGENT``: browser UA for search POSTs; empty = built-in Chrome UA."""
    web_search_prefetch_on_intent: bool = True
    """``WEB_SEARCH_PREFETCH_ON_INTENT``: run search_web in retrieve on web-search intent phrases."""
    web_skip_memory_on_intent: bool = False
    """``WEB_SKIP_MEMORY_ON_INTENT``: skip Ollama embed/retrieve when user asks for web search."""
    web_user_agent: str = "PersistentAgent-Agent/1.0 (compatible; research bot)"
    """``WEB_USER_AGENT``: User-Agent for fetch/search HTTP clients."""
    searxng_url: str = "http://127.0.0.1:8888"
    """``SEARXNG_URL``: loopback base URL for bare-metal SearXNG (primary ``search_web``)."""
    searxng_timeout_sec: float = 10.0
    """``SEARXNG_TIMEOUT_SEC``: HTTP timeout for SearXNG JSON search."""
    web_search_cache_ttl_sec: float = 86400.0
    """``WEB_SEARCH_CACHE_TTL_SEC``: SQLite search result cache TTL (default 24h)."""
    web_search_cache_path: str = ""
    """``WEB_SEARCH_CACHE_PATH``: SQLite file for search cache; empty = ~/.cache/persistent-agent/search_cache.sqlite."""
    web_browse_enabled: bool = True
    """``WEB_BROWSE_ENABLED``: allow local Playwright ``browse_web`` tool."""
    web_browse_timeout_sec: float = 30.0
    """``WEB_BROWSE_TIMEOUT_SEC``: per-navigation timeout for ``browse_web``."""
    web_browse_max_steps: int = 8
    """``WEB_BROWSE_MAX_STEPS``: max actions per browse session before forced close."""
    web_browse_max_chars: int = 28_000
    """``WEB_BROWSE_MAX_CHARS``: max extracted text returned by ``browse_web``."""
    web_browse_extra_tool_rounds: int = 3
    """``WEB_BROWSE_EXTRA_TOOL_ROUNDS``: live-turn tool-round bonus on web_intent."""

    @field_validator("llm_provider", "embed_provider", mode="before")
    @classmethod
    def _normalize_provider(cls, value: object) -> str:
        raw = str(value or "ollama").strip().lower()
        if raw in {"ollama", "openai"}:
            return raw
        _log.warning("Unknown provider %r; using ollama", value)
        return "ollama"

    @field_validator("embed_dim", mode="before")
    @classmethod
    def _positive_embed_dim(cls, value: object) -> int:
        try:
            n = int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 768
        return n if n > 0 else 768

    @model_validator(mode="after")
    def _normalize_reasoning_model(self) -> "Settings":
        if not (self.reasoning_model or "").strip():
            self.reasoning_model = "qwen3:8b"
        return self

    @model_validator(mode="after")
    def _require_openai_base_url(self) -> "Settings":
        # An OpenAI-compatible provider must name its endpoint explicitly. Falling
        # back to OLLAMA_BASE_URL would silently point at the wrong server.
        if self.llm_provider == "openai" and not (self.llm_base_url or "").strip():
            raise ValueError(
                "LLM_PROVIDER=openai requires LLM_BASE_URL "
                "(e.g. http://127.0.0.1:8000/v1)"
            )
        return self

    @field_validator(
        "rhythm_solitary_interval_min",
        mode="before",
    )
    @classmethod
    def _empty_rhythm_alias(cls, value: object) -> object:
        if value is None or value == "":
            return None
        return value

    @field_validator("llm_keep_alive", mode="before")
    @classmethod
    def _normalize_keep_alive(cls, value: str) -> str:
        raw = str(value or "").strip()
        # Negative value = keep model loaded forever (Ollama accepts "-1m").
        if re.fullmatch(r"-\d+", raw):
            return "-1m"
        # Bare positive integer (e.g. "30") is treated as minutes — Ollama requires a unit.
        if re.fullmatch(r"\d+", raw) and raw != "0":
            return f"{raw}m"
        return raw

    @property
    def database_host(self) -> str:
        """Hostname from ``database_url`` (for health scripts / start.sh probes)."""
        u = urlparse(_normalize_pg_url_for_parse(self.database_url))
        return u.hostname or "127.0.0.1"

    @property
    def database_port(self) -> int:
        """TCP port from ``database_url`` (default 5432 if omitted)."""
        u = urlparse(_normalize_pg_url_for_parse(self.database_url))
        return int(u.port) if u.port is not None else 5432

    @property
    def database_name(self) -> str:
        """First path segment of ``database_url`` (database name for dumps / seeds)."""
        u = urlparse(_normalize_pg_url_for_parse(self.database_url))
        path = (u.path or "").strip("/")
        part = path.split("/")[0] if path else ""
        return part or "agent_db"

    @property
    def database_url_plain_postgresql(self) -> str:
        """``database_url`` with ``postgresql+psycopg`` rewritten for libpq CLI tools."""
        u = self.database_url.strip()
        if u.startswith("postgresql+psycopg://"):
            return "postgresql://" + u.removeprefix("postgresql+psycopg://")
        return u

    @property
    def phylactery_restore_database_url(self) -> str:
        """Connection URL pointing at ``phylactery_restore_db`` on the same host."""
        u = urlparse(_normalize_pg_url_for_parse(self.database_url_plain_postgresql))
        userinfo = u.username or ""
        if u.password:
            userinfo = f"{userinfo}:{u.password}" if userinfo else f":{u.password}"
        auth = f"{userinfo}@" if userinfo else ""
        netloc = auth + (u.hostname or "localhost")
        if u.port:
            netloc = f"{netloc}:{u.port}"
        db = (self.phylactery_restore_db or "agent_db_restore").strip("/")
        return f"postgresql://{netloc}/{db}"


def warn_llm_num_predict_if_needed(s: Settings) -> None:
    """Log when ``LLM_NUM_PREDICT`` is too small for native thinking models."""
    if int(s.llm_num_predict) < 4096:
        _log.warning(
            "num_predict too low for a thinking model: LLM_NUM_PREDICT=%s "
            "(recommended >= 4096, default 16384)",
            s.llm_num_predict,
        )


settings = Settings()


def get_settings() -> Settings:
    """Return the process-wide settings singleton (LLM factory hooks / tests)."""
    return settings
