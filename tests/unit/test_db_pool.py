"""DB pool isolation."""

from __future__ import annotations

import src.db.pool as pool_mod
from src.db.pool import close_pool, get_checkpoint_pool, get_pool


def test_app_and_checkpoint_pools_are_distinct(monkeypatch):
    close_pool()
    monkeypatch.setattr(pool_mod.settings, "db_pool_max", 10, raising=False)
    monkeypatch.setattr(pool_mod.settings, "db_checkpoint_pool_max", 3, raising=False)
    app = get_pool()
    ckpt = get_checkpoint_pool()
    assert app is not ckpt
    close_pool()
