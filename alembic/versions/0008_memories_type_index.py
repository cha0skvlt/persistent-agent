"""Add memory_type column default and idx_memories_type index."""

from __future__ import annotations

from alembic import op

revision = "0008_memories_type_index"
down_revision = "0007_kg_agent_scope"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Column already exists for installs via 0001_init; guard for edge-case old schemas.
    op.execute("""
        ALTER TABLE memories
          ADD COLUMN IF NOT EXISTS memory_type VARCHAR(32)
          NOT NULL DEFAULT 'semantic';
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_memories_type
          ON memories (memory_type);
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_memories_type;")
