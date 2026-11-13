from __future__ import annotations

import sys
import uuid

from langchain_core.messages import HumanMessage

from src.config import settings
from src.db.pool import close_pool
from src.graph.retrieval_node import retrieve_memory
from src.memory import AgentMemory, reset_memory_singletons


def _reset_process_memory() -> None:
    """Simulate a fresh session without importing the full workflow graph."""
    close_pool()
    reset_memory_singletons()


def _delete_marker(marker: str) -> None:
    mem = AgentMemory()
    with mem.pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM memories WHERE agent_id = %s AND content ILIKE %s",
                (settings.default_agent_id, f"%{marker}%"),
            )
        conn.commit()


def main() -> int:
    """Verify that a saved memory is retrievable from a new process session."""
    marker = f"agent-e2e-{uuid.uuid4().hex}"
    fact = f"{marker}: persistent memory survives a new session."
    query = f"What do you remember about {marker}?"

    try:
        AgentMemory().save(
            fact, memory_type="semantic", agent=settings.default_agent_id
        )
        _reset_process_memory()

        result = retrieve_memory({"messages": [HumanMessage(content=query)]})
        retrieved = "\n".join(result.get("retrieved_memories") or [])
        if marker not in retrieved:
            print(
                "E2E memory retrieval failed: marker was not retrieved.",
                file=sys.stderr,
            )
            return 1

        print("E2E memory retrieval passed.")
        return 0
    finally:
        try:
            _delete_marker(marker)
        finally:
            close_pool()
            reset_memory_singletons()


if __name__ == "__main__":
    raise SystemExit(main())
