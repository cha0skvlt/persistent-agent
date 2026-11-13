"""Memory and diary: source_ref for the originating event.

``metadata.source`` records the kind of origin (user, tool, web, reflect,
background). ``source_ref`` records which event it was: a turn number, a tool
call id, a message id. It is a pointer for audit and debugging, not a proof of
attribution — the lexical prefilter remains heuristic.

Revision ID: 0028_memory_source_ref
Revises: 0027_causal_lessons_importance
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0028_memory_source_ref"
down_revision = "0027_causal_lessons_importance"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("memories", sa.Column("source_ref", sa.Text(), nullable=True))
    op.add_column("self_reflections", sa.Column("source_ref", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("self_reflections", "source_ref")
    op.drop_column("memories", "source_ref")
