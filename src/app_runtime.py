"""Process-level runtime handles — one public wipe for tests.

Individual modules keep their own ``reset_*`` / ``clear_*`` for targeted unit
tests. Pytest autouse must call ``reset_process_state_for_tests`` only — do not
reassemble the wipe in ``conftest``.
"""

from __future__ import annotations

from src.db.pool import close_pool


def clear_llm_caches() -> None:
    """Drop cached chat clients."""
    from src.llm_factory import clear_llm_cache

    clear_llm_cache()


def reset_heart_watch_state() -> None:
    """Clear heart notify / vitals / API-watch process flags."""
    try:
        from src.heart.alerts import reset_vital_alert_state
        from src.heart.api_watch import reset_api_watch_state
        from src.heart.notify import reset_notify_cooldowns

        reset_notify_cooldowns()
        reset_vital_alert_state()
        reset_api_watch_state()
    except ImportError:
        pass


def reset_cancel_tokens() -> None:
    """Clear reverie cancel tokens."""
    try:
        from src.reverie.cancel import reset_reverie_cancel_state

        reset_reverie_cancel_state()
    except ImportError:
        pass


def reset_graph_and_memory() -> None:
    """Drop compiled graph, turn semaphore, memory singletons."""
    try:
        from src.agent_runner import reset_graph_app, reset_llm_turn_semaphore

        reset_graph_app()
        reset_llm_turn_semaphore()
    except ImportError:
        pass
    try:
        from src.memory import reset_memory_singletons

        reset_memory_singletons()
    except ImportError:
        pass


def reset_identity_signer() -> None:
    try:
        from src.identity import reset_signer

        reset_signer()
    except ImportError:
        pass


def reset_main_repl_globals() -> None:
    try:
        import src.main as m

        m.app = None
        m._console = None
        m._user_prompt = None
        m._thinking_indicator = None
    except (ImportError, AttributeError):
        pass


def reset_telemetry() -> None:
    try:
        from src.obs.tracing import reset_telemetry_for_tests

        reset_telemetry_for_tests()
    except ImportError:
        pass


def reset_background_gates() -> None:
    """Solitary rhythm + background dedup clocks."""
    try:
        from src.pulse.rhythm import reset_rhythm_state

        reset_rhythm_state()
    except ImportError:
        pass
    try:
        from src.graph.bg_dedup import reset_bg_dedup

        reset_bg_dedup()
    except ImportError:
        pass


def reset_outreach_flags() -> None:
    """In-process outreach warn/fail stamps (not DB outbox)."""
    try:
        import src.heart.outreach as outreach_mod

        outreach_mod._last_send_fail_at = 0.0
        outreach_mod._warned_no_chat_id = False
    except (ImportError, AttributeError):
        pass


def reset_web_http_state() -> None:
    """Shared httpx clients + SSRF DNS cache used by web tools."""
    try:
        from src.tools import web_core as wc

        wc.reset_http_client()
        wc.reset_search_http_client()
        wc.clear_dns_public_cache()
    except (ImportError, AttributeError):
        pass


def reset_process_state_for_tests(*, close_db: bool = True) -> None:
    """Public teardown entry: clear process singletons that bleed across tests.

    Does not touch pytest monkeypatches, env, or logging handlers — fixtures own those.
    """
    clear_llm_caches()
    reset_cancel_tokens()
    reset_heart_watch_state()
    reset_identity_signer()
    reset_graph_and_memory()
    reset_main_repl_globals()
    reset_telemetry()
    reset_background_gates()
    reset_outreach_flags()
    reset_web_http_state()
    try:
        from src.embed_status import reset_embed_status

        reset_embed_status()
    except ImportError:
        pass
    try:
        from src.lessons import reset_lessons_for_tests

        reset_lessons_for_tests()
    except ImportError:
        pass
    try:
        from src.policy_knobs import reset_policy_knobs_for_tests

        reset_policy_knobs_for_tests()
    except ImportError:
        pass
    try:
        from src.obs.turn_timing import reset_turn_timing

        reset_turn_timing()
    except ImportError:
        pass
    try:
        from src.obs.deferred_job_tracker import reset_deferred_job_tracker

        reset_deferred_job_tracker()
    except ImportError:
        pass
    if close_db:
        close_pool()


__all__ = [
    "clear_llm_caches",
    "reset_background_gates",
    "reset_cancel_tokens",
    "reset_graph_and_memory",
    "reset_heart_watch_state",
    "reset_identity_signer",
    "reset_main_repl_globals",
    "reset_outreach_flags",
    "reset_process_state_for_tests",
    "reset_telemetry",
    "reset_web_http_state",
]
