"""Append-only Meta-Trace writer."""

from __future__ import annotations

from copy import deepcopy

from optiresearch.memory.schemas import MetaTrace, stable_hash
from optiresearch.storage.sqlite_store import SQLiteStore


class MetaTraceWriter:
    """Write and read immutable trace events."""

    def __init__(self, store: SQLiteStore | None = None) -> None:
        self.store = store or SQLiteStore()
        self.store.init_db()

    def write_trace(self, trace: MetaTrace) -> MetaTrace:
        candidate = self._with_content_hash(trace)
        existing_payload = self.store.get("meta_traces", candidate.trace_id)
        if existing_payload:
            existing = MetaTrace(**existing_payload)
            if self._stable_compare(existing) != self._stable_compare(candidate):
                raise ValueError(f"MetaTrace append-only conflict for trace_id={candidate.trace_id}")
            return existing
        self.store.insert_once(
            "meta_traces",
            candidate.trace_id,
            candidate,
            workspace_id=candidate.workspace_id,
            run_id=candidate.run_id,
        )
        return candidate

    def get_trace(self, trace_id: str) -> MetaTrace | None:
        payload = self.store.get("meta_traces", trace_id)
        return MetaTrace(**payload) if payload else None

    def list_traces(self, run_id: str | None = None) -> list[MetaTrace]:
        return [MetaTrace(**payload) for payload in self.store.list("meta_traces", run_id=run_id)]

    def _with_content_hash(self, trace: MetaTrace) -> MetaTrace:
        clone = trace.model_copy(deep=True)
        clone.content_hash = stable_hash(self._stable_compare(clone))
        return clone

    def _stable_compare(self, trace: MetaTrace) -> dict[str, object]:
        payload = deepcopy(trace.model_dump(mode="json"))
        payload.pop("timestamp_start", None)
        payload.pop("timestamp_end", None)
        payload.pop("content_hash", None)
        return payload
