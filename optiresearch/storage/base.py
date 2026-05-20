"""Storage protocols for the MVP."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from optiresearch.memory.schemas import ArtifactRef


class JsonStore(Protocol):
    def init_db(self) -> None: ...

    def upsert(
        self,
        table: str,
        row_id: str,
        payload: dict[str, Any],
        workspace_id: str | None = None,
        run_id: str | None = None,
    ) -> None: ...

    def get(self, table: str, row_id: str) -> dict[str, Any] | None: ...

    def list(
        self,
        table: str,
        workspace_id: str | None = None,
        run_id: str | None = None,
    ) -> list[dict[str, Any]]: ...


class ArtifactStore(Protocol):
    def register_file(
        self,
        path: str | Path,
        workspace_id: str,
        run_id: str | None,
        trace_id: str | None,
        producer: str | None,
        metadata: dict[str, Any] | None,
        metrics: dict[str, Any] | None,
    ) -> ArtifactRef: ...

    def register_json(
        self,
        payload: dict[str, Any],
        workspace_id: str,
        run_id: str | None,
        trace_id: str | None,
        producer: str | None,
        metadata: dict[str, Any] | None,
        metrics: dict[str, Any] | None,
    ) -> ArtifactRef: ...
