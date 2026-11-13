#!/usr/bin/env bash
# Agent sleep — stop resident daemons and unload Ollama models (free GPU/RAM).
# Keeps launchd plists installed; use make api-install / heart-install to wake.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
LAUNCH_AGENTS="$HOME/Library/LaunchAgents"
GUI_DOMAIN="gui/$(id -u)"

stop_launchd() {
  local label="$1"
  local plist="$LAUNCH_AGENTS/${label}.plist"
  if [[ ! -f "$plist" ]]; then
    return 0
  fi
  if launchctl print "${GUI_DOMAIN}/${label}" &>/dev/null; then
    echo "Stopping ${label}..."
    launchctl bootout "${GUI_DOMAIN}" "$plist" 2>/dev/null \
      || launchctl unload -w "$plist" 2>/dev/null \
      || true
  fi
}

echo "=== Agent sleep — releasing resources ==="

stop_launchd "cha0skvlt.persistent-agent.api"
stop_launchd "cha0skvlt.persistent-agent.heart"

pkill -f "uvicorn.run('src.api.app:app'" 2>/dev/null || true
pkill -f "src.heart.heartbeat" 2>/dev/null || true

echo "=== Unloading Ollama models ==="
if command -v curl >/dev/null 2>&1 \
  && curl -sf "http://127.0.0.1:11434/api/ps" >/dev/null 2>&1; then
  models="$(
    curl -s "http://127.0.0.1:11434/api/ps" \
      | "$REPO_ROOT/venv/bin/python3" -c "
import json, sys
data = json.load(sys.stdin)
for m in data.get('models') or []:
    name = (m.get('name') or '').strip()
    if name:
        print(name)
"
  )"
  if [[ -z "${models//[$'\n']/}" ]]; then
    echo "No models loaded."
  else
    while IFS= read -r model; do
      [[ -z "$model" ]] && continue
      echo "  ollama stop ${model}"
      ollama stop "$model" 2>/dev/null || true
    done <<< "$models"
  fi
else
  echo "Ollama not reachable — skip model unload."
fi

echo ""
echo "Agent asleep."
echo "  Wake API:  make api-install   (or: launchctl load -w ~/Library/LaunchAgents/cha0skvlt.persistent-agent.api.plist)"
echo "  Wake heart: make heart-install"
echo "  Full stack: make daemon-install"
