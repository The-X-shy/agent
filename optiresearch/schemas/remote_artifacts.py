"""Remote Artifact Manifest schema for Phase 45.

Formal Pydantic models for artifact manifests produced by remote workers
and consumed by the local ingestion pipeline.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from optiresearch.memory.schemas import StrictModel


class RemoteArtifactEntry(StrictModel):
    """A single artifact entry within a remote manifest."""

    artifact_name: str
    relative_path: str
    artifact_type: Literal[
        "metrics_json",
        "result_json",
        "report_md",
        "psf_npz",
        "reconstruction_npz",
        "trace_json",
        "manifest_json",
        "mtf_csv",
        "figure_png",
        "figure_jpg",
        "other",
    ] = "other"
    required: bool = False
    sha256: Optional[str] = None
    size_bytes: Optional[int] = None
    producer: Literal["remote_worker", "remote_cli", "ingestion"] = "remote_worker"
    source_run_id: str = ""
    source_job_id: str = ""
    evidence_role: Literal[
        "primary_metric",
        "execution_result",
        "optical_artifact",
        "reconstruction_artifact",
        "trace",
        "report",
        "auxiliary",
    ] = "auxiliary"
    metadata: dict[str, Any] = {}


class RemoteArtifactManifest(StrictModel):
    """Manifest of all artifacts produced by a remote job execution."""

    schema_version: str = "0.1"
    remote_job_id: str
    run_id: str = ""
    worker_id: str = ""
    execution_target: str = "remote_wsl"
    generated_at: str = ""
    artifacts: list[RemoteArtifactEntry] = []
    completeness: Literal["complete", "partial", "failed"] = "partial"
    missing_required_artifacts: list[str] = []
    warnings: list[str] = []
    metadata: dict[str, Any] = {}
