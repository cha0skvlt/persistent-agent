"""Pretty-printer for structured ``agent.event`` JSON (REPL + API ``make log``)."""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

from src.obs.log_format import (
    clip_words,
    fmt_kv,
    fmt_preview,
    fmt_score,
    fmt_size_chars,
)
from src.obs.logs import format_elapsed_ms

SUPPRESS = ""  # fold into active node.timing line
_NODE_FIELD_W = 20
# Channels that already draw [User]/[Agent] in the REPL/CLI UI — skip body echo.
_DIALOGUE_SKIP_CHANNELS = frozenset({"repl", "cli"})


class ConsoleFormatter(logging.Formatter):
    """Fold sub-events into ``→ node`` timing lines; frame live turns as dialogue."""

    def __init__(self) -> None:
        super().__init__(datefmt="%Y-%m-%d %H:%M:%S")
        self._use_color = bool(getattr(sys.stderr, "isatty", lambda: False)())
        self._grey = "\x1b[90m"
        self._cyan = "\x1b[36m"
        self._magenta = "\x1b[35m"
        self._bold = "\x1b[1m"
        self._reset = "\x1b[0m"
        self._node: str | None = None
        self._annots: list[str] = []
        self._in_turn = False

    def _rail(self, line: str) -> str:
        if self._in_turn:
            return f"│ {line}"
        return line

    def _fold(self, annotation: str) -> str:
        if self._node is not None:
            self._annots.append(annotation)
            return SUPPRESS
        return self._rail(f"· {annotation}")

    @staticmethod
    def _dot(category: str, detail: str) -> str:
        return f"· {category:<9} {detail}"

    @staticmethod
    def _dbg(category: str, detail: str) -> str:
        return f"  {category:<9} {detail}"

    def _role_label(self, label: str, color: str) -> str:
        if self._use_color:
            return f"{color}{self._bold}{label}{self._reset}"
        return label

    def _indent_body(self, text: str) -> str:
        lines = (text or "").splitlines() or [""]
        prefix = "│ " if self._in_turn else ""
        return "\n".join(f"{prefix}{line}" for line in lines)

    def _node_timings_note(self, timings: Any) -> str:
        if not isinstance(timings, dict) or not timings:
            return ""
        parts: list[str] = []
        for name, ms in timings.items():
            key = str(name or "").strip()
            if not key:
                continue
            parts.append(f"{key}={format_elapsed_ms(ms)}")
        if not parts:
            return ""
        return "  " + "  ".join(parts)

    def _dbg_block(self, category: str, header: str, text: str) -> str:
        """DEBUG header plus full multiline body (no word clip)."""
        head = self._dbg(category, header)
        body = text if isinstance(text, str) else str(text or "")
        if not body:
            return f"{head} ∅"
        return f"{head}\n{body}"

    def _pretty_event(self, payload: dict[str, Any]) -> str | None:
        ev = payload.get("event")
        if not isinstance(ev, str):
            return None

        # ── Live dialogue frame (Telegram / API — parallel to channel send) ─
        if ev == "turn.user":
            channel = str(payload.get("channel") or "")
            thread = str(payload.get("thread_id") or "")
            meta = fmt_kv(("channel", channel or "?"), ("thread", thread or "—"))
            self._in_turn = True
            head = f"╭ turn  {meta}"
            if channel in _DIALOGUE_SKIP_CHANNELS:
                # REPL/CLI already show [User] in the prompt UI.
                return head
            text = str(payload.get("text") or "")
            user = self._role_label("[User]", self._cyan)
            body = self._indent_body(text)
            return f"{head}\n│ {user}\n{body}"

        if ev == "turn.agent":
            channel = str(payload.get("channel") or "")
            if channel in _DIALOGUE_SKIP_CHANNELS:
                return SUPPRESS
            text = str(payload.get("text") or "")
            agent = self._role_label("[Agent]", self._magenta)
            body = self._indent_body(text)
            flag = "  (fallback)" if payload.get("fallback") else ""
            return f"│ {agent}{flag}\n{body}"

        # ── Node lifecycle ────────────────────────────────────────────────
        if ev == "node.start":
            self._node = str(payload.get("node") or "")
            self._annots = []
            return SUPPRESS

        if ev == "node.timing":
            node = str(payload.get("node") or "")
            dur = format_elapsed_ms(payload.get("elapsed_ms"))
            annots = ("  " + "  ".join(self._annots)) if self._annots else ""
            self._node = None
            self._annots = []
            return self._rail(f"→ {node:<{_NODE_FIELD_W}} {dur}{annots}")

        # ── Folded annotations (inside a node) ────────────────────────────
        if ev == "mem.search":
            top = payload.get("top_score")
            top_s = fmt_score(top) if top is not None else "—"
            return self._fold(
                f"search {fmt_kv(('hits', payload.get('hits', '?')), ('top', top_s))}"
            )

        if ev == "mem.dedup":
            preview = fmt_preview(str(payload.get("preview") or ""))
            return self._fold(
                fmt_kv(
                    ("dedup", payload.get("reason", "?")),
                    ("id", str(payload.get("old_id", ""))[:8]),
                    ("sim", payload.get("similarity")),
                    ("text", preview),
                )
            )

        if ev == "mem.save":
            preview = fmt_preview(str(payload.get("preview") or ""))
            return self._fold(f"save={preview}")

        if ev == "mem.replace":
            preview = fmt_preview(str(payload.get("preview") or ""))
            return self._fold(f"replace={preview}")

        if ev == "retrieve.result":
            self_n = payload.get("self", payload.get("self_reflections", 0))
            parts = fmt_kv(
                ("kg", payload.get("kg", 0)),
                ("mem", payload.get("mem", 0)),
                ("self", self_n),
            )
            if payload.get("emergency"):
                parts = f"emergency {parts}".strip()
            return self._fold(parts)

        if ev == "retrieve.rerank_weights":
            return self._fold(f"rerank={payload.get('rerank_weights', '?')}")

        if ev == "retrieve.skip":
            return self._fold(f"skip={payload.get('reason', '?')}")

        if ev == "retrieve.anchor":
            entity = clip_words(str(payload.get("entity") or "?"), max_words=3)
            return self._fold(fmt_kv(("anchor", entity), ("kg", payload.get("kg", 0))))

        if ev == "tool.call":
            args = payload.get("args")
            args_s = clip_words(str(args), max_words=14) if args else ""
            detail = f"tool={payload.get('name', '?')}"
            if args_s and args_s != "{}":
                detail += f" {args_s}"
            return self._fold(detail)

        if ev == "tool.result":
            name = payload.get("name", "?")
            if payload.get("empty"):
                return self._fold(f"tool={name} → ∅")
            preview = clip_words(str(payload.get("preview") or ""), max_words=18)
            size = fmt_size_chars(payload.get("chars", 0))
            return self._fold(f"tool={name} → {preview} ({size})")

        if ev == "tools.cap_cleanup":
            return self._fold(f"cap_cleanup n={payload.get('n_tools', '?')}")

        if ev == "turn.complete":
            dur = format_elapsed_ms(payload.get("elapsed_ms"))
            tools = payload.get("tools") or []
            if isinstance(tools, list) and tools:
                annots = "  ".join(f"tool={t}" for t in tools)
                line = f"→ turn {dur}  {annots}"
            else:
                line = f"→ turn {dur}"
            line += self._node_timings_note(payload.get("node_timings"))
            line = self._rail(line)
            if self._in_turn:
                self._in_turn = False
                return f"{line}\n╰"
            return line

        if ev == "htc.recover":
            path = payload.get("path")
            suffix = f"@{path}" if path else ""
            return self._fold(f"htc→{payload.get('tool', '?')}{suffix}")

        if ev == "reflect.run":
            return self._fold(
                fmt_kv(
                    ("turn", payload.get("turn_count")),
                    ("every", payload.get("reflect_every_n")),
                )
            )

        # ── Standalone INFO ─────────────────────────────────────────────────
        if ev == "reflect.skip":
            detail = fmt_kv(
                ("reason", payload.get("reason", "?")),
                ("turn", payload.get("turn_count")),
                ("every", payload.get("reflect_every_n")),
            )
            return self._dot("reflect", detail)

        if ev == "reflect.skipped":
            return self._dot("reflect", fmt_kv(("reason", payload.get("reason", "?"))))

        if ev == "reflect.blocked":
            preview = fmt_preview(str(payload.get("preview") or ""))
            return self._dot(
                "reflect",
                fmt_kv(("block", payload.get("reason", "?")), ("text", preview)),
            )

        if ev == "reflect.self":
            return self._dot(
                "reflect", f"self {fmt_preview(str(payload.get('preview') or ''))}"
            )

        if ev == "reflect.extract":
            return self._fold(
                fmt_kv(
                    ("facts", payload.get("facts", 0)),
                    ("raw", payload.get("raw_triples", 0)),
                    ("filtered", payload.get("filtered_triples", 0)),
                    ("saved", payload.get("saved_triples", 0)),
                )
            )

        # ── DEBUG ───────────────────────────────────────────────────────────
        if ev == "llm.invoke":
            path = payload.get("path", "?")
            tools = payload.get("tools", "")
            return self._dbg(
                "llm",
                fmt_kv(
                    ("path", path),
                    ("msgs", payload.get("msgs", "?")),
                    ("sys", fmt_size_chars(payload.get("sys_chars", 0))),
                    ("tools", tools),
                ),
            )

        if ev == "llm.response":
            path = payload.get("path", "?")
            tools = payload.get("tool_calls", "")
            text = str(payload.get("text") or "")
            chars = payload.get("text_chars", len(text))
            reason_chars = payload.get("reasoning_chars")
            head = fmt_kv(
                ("reply", path),
                ("text", fmt_size_chars(chars)),
                (
                    "think",
                    fmt_size_chars(reason_chars) if reason_chars is not None else None,
                ),
                ("tools", tools),
            )
            if text:
                return self._dbg_block("llm", head, text)
            return self._dbg("llm", head)

        if ev == "llm.reflect_raw":
            text = str(payload.get("text", ""))
            path = payload.get("path", "?")
            return self._dbg_block(
                "reflect",
                f"raw {path} {fmt_size_chars(len(text))}",
                text,
            )

        if ev == "retrieve.score":
            tag = payload.get("tag", "")
            preview = fmt_preview(str(payload.get("preview") or ""))
            return self._dbg(
                "mem",
                f"{tag} {fmt_score(payload.get('score', '?'))} {preview}",
            )

        if ev == "reflect.pair":
            agent_c = payload.get("agent_chars", payload.get("asst_chars", "?"))
            return self._dbg(
                "reflect",
                fmt_kv(
                    ("user", fmt_size_chars(payload.get("user_chars", 0))),
                    ("agent", fmt_size_chars(agent_c)),
                ),
            )

        if ev == "reflect.triple_blocked":
            return self._dbg(
                "reflect",
                fmt_kv(
                    ("s", clip_words(str(payload.get("subject") or "?"), max_words=3)),
                    (
                        "p",
                        clip_words(str(payload.get("predicate") or "?"), max_words=3),
                    ),
                    ("o", clip_words(str(payload.get("object") or "?"), max_words=3)),
                    ("reason", payload.get("reason", "?")),
                ),
            )

        if ev == "htc.detail":
            return self._dbg("htc", clip_words(str(payload.get("raw", ""))))

        if ev == "state.snapshot":
            return self._dbg("state", str(payload.get("keys", "")))

        # Compact fallback for any other structured event
        skip = frozenset({"event", "ts_ms"})
        extra = fmt_kv(
            *((k, payload[k]) for k in sorted(payload) if k not in skip),
        )
        if extra:
            return self._dot(ev.replace(".", " "), extra)
        return self._dot(ev.replace(".", " "), "—")

    def format(self, record: logging.LogRecord) -> str:
        msg = record.getMessage()
        pretty: str | None = None
        if isinstance(msg, str) and msg.startswith("{") and msg.endswith("}"):
            try:
                payload = json.loads(msg)
                if isinstance(payload, dict):
                    pretty = self._pretty_event(payload)
            except Exception:
                pretty = None

        if pretty == SUPPRESS:
            return SUPPRESS

        message_out = pretty if pretty is not None else msg
        ts = self.formatTime(record, self.datefmt)
        level = record.levelname
        col = {
            "BOOT": "BOOT ",
            "INFO": "INFO ",
            "DEBUG": "DEBUG",
            "ERROR": "ERROR",
        }.get(level, f"{level:<5}")
        # Multiline dialogue: stamp only the first line; keep │ rail aligned.
        parts = str(message_out).splitlines() or [""]
        stamped = [f"{ts} | {col} | {parts[0]}"]
        pad = " " * len(ts)
        for cont in parts[1:]:
            stamped.append(f"{pad} |       | {cont}")
        line = "\n".join(stamped)
        if self._use_color:
            # Role colors already in message — do not wash with dim grey.
            if "\x1b[" in str(message_out):
                return line
            if level == "DEBUG":
                return f"\x1b[36m\x1b[2m{line}\x1b[0m"
            if level in {"INFO", "BOOT"}:
                return f"{self._grey}\x1b[2m{line}{self._reset}"
            return f"{self._grey}{line}{self._reset}"
        return line


def _yn(val: Any) -> str:
    return "yes" if val else "no"
