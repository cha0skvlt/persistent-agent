"""Normalize legacy agent_id values on memories to default \"agent\".

Legacy REPL/tests used test_agent / miqu_witch; production default is DEFAULT_AGENT_ID=agent.
"""

from __future__ import annotations

from alembic import op

revision = "0005_memory_legacy_agent_ids"
down_revision = "0004_kg_entities"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        UPDATE memories
        SET agent_id = 'agent'
        WHERE agent_id IN ('test_agent', 'miqu_witch')
        """)


def downgrade() -> None:
    # Irreversible: cannot distinguish rows that were migrated vs. originally \"agent\".
    pass
