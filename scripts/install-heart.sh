#!/usr/bin/env bash
# Install the persistent-agent heart daemon via launchd.
# Must be run from the repo root directory.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
PLIST_SRC="$REPO_ROOT/launchd/cha0skvlt.persistent-agent.heart.plist"
PLIST_DEST="$HOME/Library/LaunchAgents/cha0skvlt.persistent-agent.heart.plist"
LOG_PATH="$HOME/Library/Logs/agent-heart.log"
LAUNCH_AGENTS_DIR="$HOME/Library/LaunchAgents"

# Strip optional surrounding quotes (``.env`` must be unquoted for pydantic-settings too).
_strip_env_value() {
    local v="${1#"${1%%[![:space:]]*}"}"
    v="${v%"${v##*[![:space:]]}"}"
    if [[ "$v" == \"*\" && "$v" == *\" ]]; then
        v="${v:1:${#v}-2}"
    elif [[ "$v" == \'*\' && "$v" == *\' ]]; then
        v="${v:1:${#v}-2}"
    fi
    printf '%s' "$v"
}

# Resolve DATABASE_URL: env → .env → default. Written into launchd plist for heart process.
if [ -z "${DATABASE_URL:-}" ] && [ -f "$REPO_ROOT/.env" ]; then
    DATABASE_URL="$(_strip_env_value "$(grep -E '^DATABASE_URL=' "$REPO_ROOT/.env" | head -1 | cut -d= -f2-)")"
fi
DATABASE_URL="$(_strip_env_value "${DATABASE_URL:-postgresql://localhost:5432/agent_db}")"

# launchd requires absolute paths — no bare ``python3`` or ``/usr/bin/env``.
VENV_PYTHON="$REPO_ROOT/venv/bin/python3"
if [ ! -x "$VENV_PYTHON" ]; then
    echo "ERROR: venv not found at $VENV_PYTHON. Run: python3 -m venv venv && pip install -r requirements.txt" >&2
    exit 1
fi
VENV_PYTHON="$(cd "$(dirname "$VENV_PYTHON")" && pwd -P)/$(basename "$VENV_PYTHON")"
if [[ "$VENV_PYTHON" != /* ]]; then
    echo "ERROR: resolved python path is not absolute: $VENV_PYTHON" >&2
    exit 1
fi

if ! "$VENV_PYTHON" -c "import psycopg" 2>/dev/null; then
    echo "ERROR: psycopg import failed (launchd needs bundled libpq)." >&2
    echo "Run: $VENV_PYTHON -m pip install -r $REPO_ROOT/requirements.txt" >&2
    exit 1
fi

mkdir -p "$LAUNCH_AGENTS_DIR"
mkdir -p "$(dirname "$LOG_PATH")"

# Expand placeholders in the plist template (paths must stay absolute).
# Python replace — sed treats ``&`` in DATABASE_URL as "matched text".
"$VENV_PYTHON" - "$PLIST_SRC" "$PLIST_DEST" "$VENV_PYTHON" "$REPO_ROOT" "$LOG_PATH" "$DATABASE_URL" <<'PY'
import pathlib
import sys

src, dest, venv_py, repo, log_path, db_url = sys.argv[1:7]
text = pathlib.Path(src).read_text(encoding="utf-8")
for old, new in (
    ("AGENT_VENV_PYTHON", venv_py),
    ("AGENT_REPO_ROOT", repo),
    ("AGENT_LOG_PATH", log_path),
    ("AGENT_DATABASE_URL", db_url),
):
    text = text.replace(old, new)
pathlib.Path(dest).write_text(text, encoding="utf-8")
PY

echo "Plist written to $PLIST_DEST"
echo "Program: $VENV_PYTHON -m src.heart.heartbeat"
echo "WorkingDirectory: $REPO_ROOT"

# Unload if already loaded (idempotent reinstall).
if launchctl list | grep -q "cha0skvlt.persistent-agent.heart" 2>/dev/null; then
    echo "Unloading existing heart service..."
    launchctl unload -w "$PLIST_DEST" 2>/dev/null || true
fi

launchctl load -w "$PLIST_DEST"
echo "Heart daemon loaded."

sleep 2
if launchctl list | grep -q "cha0skvlt.persistent-agent.heart"; then
    echo "OK: $(launchctl list | grep cha0skvlt.persistent-agent.heart)"
else
    echo "WARNING: service not visible in launchctl list yet (may still be starting)."
    echo "Check log: tail -20 $LOG_PATH"
fi
