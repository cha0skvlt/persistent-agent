"""Historical no-op: the legacy agent-id rename never ran as intended.

This revision was generated during a rename of the database scope id. The
generated statements collapsed into ``SET agent_id = 'agent' WHERE agent_id =
'agent'`` and, worse, ended with a ``DELETE FROM agent_identity`` that could
destroy the Ed25519 key row on an existing database.

The revision id and ``down_revision`` are kept so the chain 0001–0027 stays
intact. The body is intentionally empty: a database that already applied this
revision is unaffected, and a fresh database has nothing to rename.

If an existing database lost its identity row to the old body, restore it from a
backup. Without a backup, create a new identity with ``make init-identity`` —
signatures made with the old key will not verify against the new one.
"""

from __future__ import annotations

revision = "0014_agent_id_agent_to_agent"
down_revision = "0013_hot_perf_indexes"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """No-op. See the module docstring."""


def downgrade() -> None:
    """No-op. See the module docstring."""
