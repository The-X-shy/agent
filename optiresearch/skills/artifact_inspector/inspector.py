"""Artifact inspection helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from optiresearch.memory.schemas import ArtifactRef, compute_file_sha256
from optiresearch.storage.file_artifact_store import FileArtifactStore


class ArtifactInspector:
    """Inspect registered artifact files without changing them."""

    def __init__(self, artifact_store: FileArtifactStore | None = None) -> None:
        self.artifact_store = artifact_store or FileArtifactStore()

    def inspect_artifact(self, artifact_ref: ArtifactRef) -> dict[str, Any]:
        path = self.artifact_store.resolve_uri(artifact_ref.uri)
        artifact_type = artifact_ref.metadata.get("artifact_type", "unknown")
        base = {
            "artifact_id": artifact_ref.artifact_id,
            "artifact_type": artifact_type,
            "uri": artifact_ref.uri,
            "content_hash": artifact_ref.content_hash,
            "file_exists": path.exists(),
        }
        if not path.exists():
            return {**base, "error": "artifact file is missing"}
        if artifact_type == "metrics":
            return {**base, **self._inspect_metrics(path)}
        if artifact_type == "psf_cube":
            return {**base, **self._inspect_npz(path)}
        if artifact_type == "mtf_curve":
            return {**base, **self._inspect_csv(path)}
        if artifact_type == "figure":
            return {**base, "file_size_bytes": path.stat().st_size, "sha256": compute_file_sha256(path)}
        return {**base, "file_size_bytes": path.stat().st_size, "sha256": compute_file_sha256(path)}

    def _inspect_metrics(self, path: Path) -> dict[str, Any]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        numeric = {key: value for key, value in payload.items() if isinstance(value, (int, float))}
        return {
            "metric_names": sorted(numeric),
            "metrics": numeric,
            "metric_count": len(numeric),
        }

    def _inspect_npz(self, path: Path) -> dict[str, Any]:
        arrays: dict[str, list[int]] = {}
        with np.load(path) as data:
            for name in data.files:
                arrays[name] = list(data[name].shape)
        return {"arrays": arrays, "array_count": len(arrays)}

    def _inspect_csv(self, path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle))
        columns = rows[0] if rows else []
        return {"columns": columns, "row_count": max(len(rows) - 1, 0), "column_count": len(columns)}
