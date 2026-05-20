"""RunMemory access helpers."""

from __future__ import annotations

from optiresearch.memory.schemas import RunMemory
from optiresearch.storage.sqlite_store import SQLiteStore


class RunMemoryStore:
    def __init__(self, store: SQLiteStore | None = None) -> None:
        self.store = store or SQLiteStore()

    def latest(self, run_id: str) -> RunMemory | None:
        rows = self.store.list("run_memories", run_id=run_id)
        if not rows:
            return None
        return RunMemory(**max(rows, key=lambda row: row["version"]))
