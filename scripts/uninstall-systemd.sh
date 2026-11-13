#!/usr/bin/env bash
# Remove the persistent-agent systemd services.
set -euo pipefail

UNIT_DIR="${UNIT_DIR:-/etc/systemd/system}"

SUDO=""
if [[ "$(id -u)" -ne 0 ]]; then SUDO="sudo"; fi

for unit in persistent-agent-api persistent-agent-heart; do
  $SUDO systemctl disable --now "$unit.service" 2>/dev/null || true
  $SUDO rm -f "$UNIT_DIR/$unit.service"
  echo "removed $UNIT_DIR/$unit.service"
done

$SUDO systemctl daemon-reload
