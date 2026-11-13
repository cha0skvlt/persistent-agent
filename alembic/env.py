from __future__ import annotations

import sys
from logging.config import fileConfig
from pathlib import Path

from sqlalchemy import create_engine, text

from alembic import context

# Repo root on sys.path so ``from src.config`` works without ``PYTHONPATH=.``.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)


def get_url() -> str:
    from src.config import settings

    url = settings.database_url
    # Force psycopg (v3) driver; repo uses psycopg, not psycopg2.
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


def run_migrations_online() -> None:
    connectable = create_engine(get_url(), future=True)

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=None)

        with context.begin_transaction():
            # KISS: schema is defined via raw SQL, no ORM metadata.
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            context.run_migrations()


run_migrations_online()
