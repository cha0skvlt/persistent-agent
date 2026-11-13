"""Outbox footer_text — same → turn line as live Telegram replies.

Revision ID: 0022_outbox_footer
Revises: 0021_intents
"""

from __future__ import annotations

from alembic import op

revision = "0022_outbox_footer"
down_revision = "0021_intents"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE outbox ADD COLUMN IF NOT EXISTS footer_text TEXT")


def downgrade() -> None:
    op.execute("ALTER TABLE outbox DROP COLUMN IF EXISTS footer_text")
