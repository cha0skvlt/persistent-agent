#!/usr/bin/env bash
# Install persistent-agent as systemd services (API + Heart).
# Linux counterpart of the macOS launchd bundle.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
VENV_PY="$ROOT/venv/bin/python3"
UNIT_DIR="${UNIT_DIR:-/etc/systemd/system}"

[[ -x "$VENV_PY" ]] || {
  echo "ERROR: $VENV_PY missing — run ./install.sh first" >&2
  exit 1
}
command -v systemctl >/dev/null 2>&1 || {
  echo "ERROR: systemd not found on this host" >&2
  exit 1
}

SUDO=""
if [[ "$(id -u)" -ne 0 ]]; then SUDO="sudo"; fi

for unit in persistent-agent-api persistent-agent-heart; do
  sed -e "s|@REPO_ROOT@|$ROOT|g" -e "s|@VENV_PYTHON@|$VENV_PY|g" \
    "$ROOT/systemd/$unit.service" \
    | $SUDO tee "$UNIT_DIR/$unit.service" >/dev/null
  echo "installed $UNIT_DIR/$unit.service"
done

$SUDO systemctl daemon-reload
$SUDO systemctl enable --now persistent-agent-api.service
$SUDO systemctl enable --now persistent-agent-heart.service
$SUDO systemctl --no-pager --lines=0 status \
  persistent-agent-api.service persistent-agent-heart.service || true
