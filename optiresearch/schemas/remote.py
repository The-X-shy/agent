"""Schemas for remote worker execution."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import Field, field_validator

from optiresearch.memory.schemas import StrictModel


RemoteJobType = Literal[
    "hsi_reconstruction",
    "hsi_matrix",
    "codesign_loop",
    "deeplens_source_smoke",
    "native_optimization_probe",
    "native_optimization_inspection",
    "deeplens_surface_optimization_probe",
    "deeplens_lensfile_optimization_probe",
    "native_hsi_codesign",
    "native_hsi_reconstruction_codesign",
    "deeplens_waveoptics_probe",
    "native_waveoptics_hsi_codesign",
    "autonomous_loop",
]

RemoteJobStatus = Literal["pending", "running", "succeeded", "failed", "timeout", "skipped"]


class RemoteWorkerSpec(StrictModel):
    worker_id: str = Field(min_length=1)
    host: str = Field(min_length=1)
    port: int = Field(default=22, ge=1, le=65535)
    username: str = Field(min_length=1)
    ssh_key_path: Optional[str] = None
    remote_project_dir: str = Field(min_length=1)
    remote_workspace_dir: str = Field(min_length=1)
    python_executable: str = Field(min_length=1)
    environment_name: Optional[str] = None
    max_runtime_seconds: int = Field(default=3600, ge=1)
    backend_tags: list[str] = Field(default_factory=list)
    capabilities: dict[str, Any] = Field(default_factory=dict)

    @field_validator("remote_project_dir", "remote_workspace_dir", "python_executable")
    @classmethod
    def remote_paths_are_absolute(cls, value: str) -> str:
        if not value.startswith("/"):
            raise ValueError("remote paths must be absolute")
        return value


class RemoteJobSpec(StrictModel):
    job_id: str = Field(min_length=1)
    job_type: RemoteJobType
    objective: str = Field(min_length=1)
    cli_args: dict[str, Any] = Field(default_factory=dict)
    input_artifacts: list[str] = Field(default_factory=list)
    expected_outputs: list[str] = Field(default_factory=list)
    timeout_seconds: int = Field(ge=1)
    evidence_policy: dict[str, Any] = Field(default_factory=dict)


class RemoteJobResult(StrictModel):
    job_id: str = Field(min_length=1)
    status: RemoteJobStatus
    remote_run_id: Optional[str]
    started_at: str
    finished_at: str
    command: list[str] = Field(default_factory=list)
    stdout_path: str
    stderr_path: str
    remote_output_dir: str
    local_output_dir: str
    artifact_manifest: dict[str, Any] = Field(default_factory=dict)
    metrics_summary: dict[str, Any] = Field(default_factory=dict)
    error_code: Optional[str]
    caveats: list[str] = Field(default_factory=list)
