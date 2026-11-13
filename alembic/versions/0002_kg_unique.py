from __future__ import annotations

from alembic import op

revision = "0002_kg_unique"
down_revision = "0001_init"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Use md5(subject|predicate|object) as fixed-size uniqueness key.
    op.execute("""
        ALTER TABLE knowledge_graph
          ADD COLUMN IF NOT EXISTS triple_hash CHAR(32);
        """)
    op.execute("""
        UPDATE knowledge_graph
        SET triple_hash = md5(subject || '|' || predicate || '|' || object)
        WHERE triple_hash IS NULL;
        """)
    op.execute("""
        DELETE FROM knowledge_graph a
        USING knowledge_graph b
        WHERE a.id > b.id
          AND a.triple_hash = b.triple_hash;
        """)
    op.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS uq_knowledge_graph_triple_hash
          ON knowledge_graph (triple_hash);
        """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS uq_knowledge_graph_triple_hash;")
    op.execute("ALTER TABLE knowledge_graph DROP COLUMN IF EXISTS triple_hash;")
