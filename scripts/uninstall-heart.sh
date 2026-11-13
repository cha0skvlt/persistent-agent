#!/usr/bin/env bash
# Uninstall the persistent-agent heart daemon.
set -euo pipefail

PLIST_DEST="$HOME/Library/LaunchAgents/cha0skvlt.persistent-agent.heart.plist"

if [ ! -f "$PLIST_DEST" ]; then
    echo "Plist not found at $PLIST_DEST — nothing to uninstall."
    exit 0
fi

launchctl unload -w "$PLIST_DEST" 2>/dev/null || true
echo "Heart daemon unloaded."

rm -f "$PLIST_DEST"
echo "Plist removed: $PLIST_DEST"
echo "Log file retained at $HOME/Library/Logs/agent-heart.log"
