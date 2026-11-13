"""Copytruncate rotation for launchd-redirected log files.

launchd holds the log fd open in append mode, so rename-based rotation would
leave writers on the old inode. Copy → truncate keeps the inode; append-mode
writers continue at the new EOF. A short race window between copy and truncate
may drop a few lines — acceptable for ops logs.

Heart calls ``rotate_agent_logs_if_needed`` once per tick (cheap stat).
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from src.config import settings
from src.obs.logs import log_event

_log = logging.getLogger(__name__)

_ROTATE_GLOB = "agent-*.log"


def rotate_if_needed(path: Path, *, max_bytes: int, backups: int) -> bool:
    """Rotate one file when it exceeds ``max_bytes``. Returns True if rotated."""
    try:
        if not path.is_file() or path.stat().st_size <= max_bytes:
            return False
        for i in range(backups - 1, 0, -1):
            src = path.with_name(f"{path.name}.{i}")
            if src.exists():
                shutil.move(str(src), str(path.with_name(f"{path.name}.{i + 1}")))
        if backups > 0:
            shutil.copy2(str(path), str(path.with_name(f"{path.name}.1")))
        with path.open("r+") as fh:
            fh.truncate(0)
        return True
    except Exception as exc:
        _log.warning("log rotate failed path=%s: %s", path, exc)
        return False


# Rotated by its own in-process RotatingFileHandler (debug_context) — a second
# copytruncate here would race the handler's rename-based rollover.
_ROTATE_SKIP = frozenset({"agent-context.log"})


def rotate_agent_logs_if_needed() -> None:
    """Rotate every ``agent-*.log`` in ~/Library/Logs above the size cap."""
    max_bytes = max(1, int(settings.log_rotate_max_mb)) * 1024 * 1024
    backups = max(0, int(settings.log_rotate_backups))
    logs_dir = Path.home() / "Library" / "Logs"
    for path in sorted(logs_dir.glob(_ROTATE_GLOB)):
        if path.name in _ROTATE_SKIP:
            continue
        if rotate_if_needed(path, max_bytes=max_bytes, backups=backups):
            log_event("log.rotated", path=str(path), max_mb=settings.log_rotate_max_mb)


__all__ = ["rotate_agent_logs_if_needed", "rotate_if_needed"]
