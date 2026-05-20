"""Local filesystem artifact store."""

from __future__ import annotations

import json
import mimetypes
import os
import shutil
from pathlib import Path
from typing import Any

from optiresearch.memory.schemas import ArtifactRef, compute_file_sha256, make_artifact_id
from optiresearch.storage.sqlite_store import SQLiteStore


class FileArtifactStore:
    """Register files and JSON payloads under a deterministic local artifact root."""

    def __init__(self, root: str | Path | None = None, store: SQLiteStore | None = None) -> None:
        self.root = Path(root or os.getenv("OPTIRESEARCH_ARTIFACT_ROOT", "./workspace/artifacts"))
        self.store = store or SQLiteStore()
        self.root.mkdir(parents=True, exist_ok=True)
        self.store.init_db()

    def register_file(
        self,
        path: str | Path,
        workspace_id: str,
        run_id: str | None,
        trace_id: str | None,
        producer: str | None,
        metadata: dict[str, Any] | None,
        metrics: dict[str, Any] | None,
    ) -> ArtifactRef:
        source = Path(path)
        content_hash = compute_file_sha256(source)
        artifact_id = make_artifact_id(workspace_id, run_id, trace_id, content_hash, producer, source.name)
        suffix = source.suffix or ".bin"
        dest = self._run_dir(workspace_id, run_id) / f"{artifact_id}{suffix}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists() or compute_file_sha256(dest) != content_hash:
            shutil.copy2(source, dest)
        enriched_metadata = self._enrich_metadata(source, metadata or {}, metrics or {}, producer)
        parsed_metrics = metrics or {}
        if not parsed_metrics and enriched_metadata["artifact_type"] == "metrics":
            parsed_metrics = self._parse_numeric_metrics(source)
            enriched_metadata["metric_names"] = sorted(parsed_metrics)
        ref = ArtifactRef(
            artifact_id=artifact_id,
            workspace_id=workspace_id,
            run_id=run_id,
            trace_id=trace_id,
            uri=self._safe_uri(dest),
            mime=mimetypes.guess_type(source.name)[0],
            content_hash=content_hash,
            producer=producer,
            metadata=enriched_metadata,
            metrics=parsed_metrics,
        )
        self.store.upsert("artifacts", ref.artifact_id, ref, workspace_id=workspace_id, run_id=run_id)
        return ref

    def register_json(
        self,
        payload: dict[str, Any],
        workspace_id: str,
        run_id: str | None,
        trace_id: str | None,
        producer: str | None,
        metadata: dict[str, Any] | None,
        metrics: dict[str, Any] | None,
    ) -> ArtifactRef:
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
        content_hash = self._bytes_sha256(raw)
        artifact_id = make_artifact_id(workspace_id, run_id, trace_id, content_hash, producer, "payload.json")
        dest = self._run_dir(workspace_id, run_id) / f"{artifact_id}.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists() or dest.read_bytes() != raw:
            dest.write_bytes(raw)
        enriched_metadata = self._enrich_metadata(Path("payload.json"), metadata or {}, metrics or {}, producer)
        ref = ArtifactRef(
            artifact_id=artifact_id,
            workspace_id=workspace_id,
            run_id=run_id,
            trace_id=trace_id,
            uri=self._safe_uri(dest),
            mime="application/json",
            content_hash=content_hash,
            producer=producer,
            metadata=enriched_metadata,
            metrics=metrics or {},
        )
        self.store.upsert("artifacts", ref.artifact_id, ref, workspace_id=workspace_id, run_id=run_id)
        return ref

    def get_artifact(self, artifact_id: str) -> ArtifactRef | None:
        payload = self.store.get("artifacts", artifact_id)
        return ArtifactRef(**payload) if payload else None

    def list_artifacts(self, run_id: str | None = None) -> list[ArtifactRef]:
        return [ArtifactRef(**payload) for payload in self.store.list("artifacts", run_id=run_id)]

    def resolve_uri(self, uri: str) -> Path:
        return self.root.parent / uri

    def _enrich_metadata(
        self,
        source: Path,
        metadata: dict[str, Any],
        metrics: dict[str, Any],
        producer: str | None,
    ) -> dict[str, Any]:
        enriched = dict(metadata)
        filename = enriched.get("filename", source.name)
        enriched.setdefault("filename", filename)
        enriched.setdefault("artifact_type", self._artifact_type(str(filename)))
        enriched.setdefault("metric_names", sorted(k for k, v in metrics.items() if isinstance(v, (int, float))))
        enriched.setdefault("producer_skill_id", enriched.get("producer_skill_id") or self._producer_skill_id(producer))
        enriched.setdefault("producer_skill_version", enriched.get("producer_skill_version") or "0.1.0")
        return enriched

    def _parse_numeric_metrics(self, path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return {key: value for key, value in payload.items() if isinstance(value, (int, float))}

    def _artifact_type(self, filename: str) -> str:
        lower = filename.lower()
        if lower.endswith(".npz") or "psf_cube" in lower:
            return "psf_cube"
        if lower.endswith(".csv") or "mtf" in lower:
            return "mtf_curve"
        if lower.endswith(".json") and "metrics" in lower:
            return "metrics"
        if lower.endswith(".json") and "manifest" in lower:
            return "manifest"
        if lower.endswith((".png", ".jpg", ".jpeg")):
            return "figure"
        return "unknown"

    def _producer_skill_id(self, producer: str | None) -> str | None:
        if producer and "DeepLens" in producer:
            return "deeplens-adapter"
        return None

    def _run_dir(self, workspace_id: str, run_id: str | None) -> Path:
        return self.root / workspace_id / (run_id or "unscoped")

    def _safe_uri(self, path: Path) -> str:
        return path.relative_to(self.root.parent).as_posix()

    @staticmethod
    def _bytes_sha256(raw: bytes) -> str:
        import hashlib

        return hashlib.sha256(raw).hexdigest()
