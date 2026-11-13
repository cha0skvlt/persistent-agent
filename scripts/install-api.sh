#!/usr/bin/env bash
# Install the persistent-agent API daemon via launchd (ARCH-daemon).
# Must be run from the repo root directory.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
PLIST_SRC="$REPO_ROOT/launchd/cha0skvlt.persistent-agent.api.plist"
PLIST_DEST="$HOME/Library/LaunchAgents/cha0skvlt.persistent-agent.api.plist"
LOG_PATH="$HOME/Library/Logs/agent-api.log"
LAUNCH_AGENTS_DIR="$HOME/Library/LaunchAgents"

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

if ! "$VENV_PYTHON" -c "import uvicorn" 2>/dev/null; then
    echo "ERROR: uvicorn import failed." >&2
    echo "Run: $VENV_PYTHON -m pip install -r $REPO_ROOT/requirements.txt" >&2
    exit 1
fi

mkdir -p "$LAUNCH_AGENTS_DIR"
mkdir -p "$(dirname "$LOG_PATH")"

# Expand placeholders in the plist template (paths must stay absolute).
sed \
    -e "s|AGENT_VENV_PYTHON|$VENV_PYTHON|g" \
    -e "s|AGENT_REPO_ROOT|$REPO_ROOT|g" \
    -e "s|AGENT_LOG_PATH|$LOG_PATH|g" \
    "$PLIST_SRC" > "$PLIST_DEST"

echo "Plist written to $PLIST_DEST"
echo "Program: $VENV_PYTHON -c \"import uvicorn; ... uvicorn.run('src.api.app:app', ...)\""
echo "WorkingDirectory: $REPO_ROOT"
echo "Log: $LOG_PATH"

# Unload if already loaded (idempotent reinstall).
if launchctl list | grep -q "cha0skvlt.persistent-agent.api" 2>/dev/null; then
    echo "Unloading existing api service..."
    launchctl unload -w "$PLIST_DEST" 2>/dev/null || true
fi

launchctl load -w "$PLIST_DEST"
echo "API daemon loaded."

sleep 2
if launchctl list | grep -q "cha0skvlt.persistent-agent.api"; then
    echo "OK: $(launchctl list | grep cha0skvlt.persistent-agent.api)"
else
    echo "WARNING: service not visible in launchctl list yet (may still be starting)."
    echo "Check log: tail -20 $LOG_PATH"
fi
