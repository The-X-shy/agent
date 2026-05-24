"""Remote artifact ingestion with sha256 verification for Phase 45."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RemoteArtifactIngestionResult:
    remote_job_id: str = ""
    run_id: str = ""
    manifest_path: str = ""
    completeness: str = "partial"
    ingested_artifacts: list[dict[str, Any]] = field(default_factory=list)
    missing_required_artifacts: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    artifact_ids: list[str] = field(default_factory=list)
    primary_metric_artifact_id: str = ""
    execution_result_artifact_id: str = ""


def ingest_remote_artifact_manifest(
    manifest_path: str | Path,
    workspace_id: str = "default",
) -> RemoteArtifactIngestionResult:
    """Read a remote artifact manifest, verify sha256, and register in ArtifactStore."""
    from optiresearch.storage.file_artifact_store import FileArtifactStore

    manifest_path = Path(manifest_path)
    result = RemoteArtifactIngestionResult(manifest_path=str(manifest_path))

    if not manifest_path.exists():
        result.errors.append(f"Manifest not found: {manifest_path}")
        result.warnings.append("Manifest missing — cannot index artifacts")
        return result

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as e:
        result.errors.append(f"Failed to parse manifest: {e}")
        return result

    result.remote_job_id = manifest.get("remote_job_id", "")
    result.run_id = manifest.get("run_id", "")
    result.completeness = manifest.get("completeness", "partial")
    result.missing_required_artifacts = manifest.get("missing_required_artifacts", [])

    store = FileArtifactStore()
    manifest_dir = manifest_path.parent

    for entry in manifest.get("artifacts", []):
        rel_path = entry.get("relative_path", entry.get("artifact_name", ""))
        artifact_path = _resolve_path(manifest_dir, rel_path)

        if not artifact_path or not artifact_path.exists():
            if entry.get("required"):
                result.missing_required_artifacts.append(entry.get("artifact_name", rel_path))
                result.errors.append(f"Required artifact missing: {entry.get('artifact_name', rel_path)}")
            else:
                result.warnings.append(f"Optional artifact not found: {entry.get('artifact_name', rel_path)}")
            continue

        # Verify sha256
        expected_sha = entry.get("sha256", "")
        if expected_sha:
            actual_sha = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
            if actual_sha != expected_sha:
                result.errors.append(
                    f"SHA256 mismatch for {entry.get('artifact_name', rel_path)}: "
                    f"expected {expected_sha[:16]}..., got {actual_sha[:16]}..."
                )
                continue

        # Register in ArtifactStore
        try:
            artifact_id = store.register_file(
                str(artifact_path),
                workspace_id=workspace_id,
                run_id=result.run_id,
                trace_id=result.remote_job_id,
                producer="remote-worker",
                metadata={
                    "source": "remote_wsl",
                    "remote_job_id": result.remote_job_id,
                    "remote_worker_id": manifest.get("worker_id", ""),
                    "execution_target": manifest.get("execution_target", "remote_wsl"),
                    "evidence_role": entry.get("evidence_role", "auxiliary"),
                    "artifact_type": entry.get("artifact_type", "other"),
                    "manifest_id": result.remote_job_id,
                    "filename": entry.get("artifact_name", rel_path),
                },
                metrics=entry.get("metadata", {}).get("metrics", {}),
            )
        except Exception as e:
            result.errors.append(f"Failed to register {entry.get('artifact_name', rel_path)}: {e}")
            continue

        ingested = {
            "artifact_id": artifact_id,
            "artifact_name": entry.get("artifact_name", rel_path),
            "artifact_type": entry.get("artifact_type", "other"),
            "local_path": str(artifact_path),
            "sha256": expected_sha,
            "evidence_role": entry.get("evidence_role", "auxiliary"),
        }
        result.ingested_artifacts.append(ingested)
        result.artifact_ids.append(artifact_id)

        if entry.get("evidence_role") == "primary_metric" and not result.primary_metric_artifact_id:
            result.primary_metric_artifact_id = artifact_id
        if entry.get("evidence_role") == "execution_result" and not result.execution_result_artifact_id:
            result.execution_result_artifact_id = artifact_id

    return result


def _resolve_path(manifest_dir: Path, rel_path: str) -> Path | None:
    direct = manifest_dir / rel_path
    if direct.exists():
        return direct
    # Try nested under outputs/
    nested = manifest_dir / "outputs" / rel_path
    if nested.exists():
        return nested
    # Try with manifest_dir name as parent
    double_nested = manifest_dir / manifest_dir.name / rel_path
    if double_nested.exists():
        return double_nested
    double_nested2 = manifest_dir / "outputs" / manifest_dir.name / rel_path
    if double_nested2.exists():
        return double_nested2
    return None


def validate_remote_artifact_manifest(manifest_path: str | Path) -> dict[str, Any]:
    """Validate a remote artifact manifest and return issues."""
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        return {"valid": False, "errors": [f"Manifest not found: {manifest_path}"]}

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as e:
        return {"valid": False, "errors": [f"Failed to parse: {e}"]}

    errors: list[str] = []
    warnings: list[str] = []

    version = manifest.get("schema_version", "")
    if not version:
        errors.append("Missing schema_version")

    if not manifest.get("remote_job_id"):
        errors.append("Missing remote_job_id")

    artifacts = manifest.get("artifacts", [])
    if not artifacts:
        warnings.append("No artifacts listed in manifest")

    for i, entry in enumerate(artifacts):
        name = entry.get("artifact_name", f"index_{i}")
        if not entry.get("relative_path") and not entry.get("path"):
            errors.append(f"Artifact '{name}' missing relative_path")

    completeness = manifest.get("completeness", "")
    if completeness not in ("complete", "partial", "failed"):
        errors.append(f"Unknown completeness: {completeness}")

    missing = manifest.get("missing_required_artifacts", [])
    if missing and completeness == "complete":
        errors.append("Completeness is 'complete' but missing_required_artifacts is non-empty")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "schema_version": version,
        "remote_job_id": manifest.get("remote_job_id", ""),
        "completeness": completeness,
        "artifact_count": len(artifacts),
        "missing_required_artifacts": missing,
    }
