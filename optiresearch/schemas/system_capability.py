"""System capability registry schema for Phase 68."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from optiresearch.memory.schemas import StrictModel


class SystemCapabilityEntry(StrictModel):
    """Unified capability record for any system component."""

    capability_id: str
    capability_type: Literal[
        "handler", "skill", "design", "backend", "dataset",
        "remote_worker", "artifact", "report", "benchmark", "claim_policy",
    ]
    name: str = ""
    enabled: bool = True
    maturity_level: Literal[
        "experimental", "validated_local", "validated_remote", "benchmarked", "production_ready",
    ] = "experimental"
    supported_execution_modes: list[str] = []
    evidence_level: str = "unsupported"
    max_claim_ceiling: str = "unsupported"
    synthetic_only: bool = False
    real_data_required: bool = False
    native_backend_required: bool = False
    physical_backend: bool = False
    supports_remote: bool = False
    requires_remote: bool = False
    requires_deeplens: bool = False
    requires_wsl: bool = False
    requires_gpu: bool = False
    required_env_vars: list[str] = []
    required_files: list[str] = []
    output_contract_id: str = ""
    artifact_contract_id: str = ""
    report_contract_id: str = ""
    known_limitations: list[str] = []
    blocked_claims: list[str] = []
    safe_wording: str = ""
    owner_module: str = ""
    tests: list[str] = []
    docs: list[str] = []


class SystemCapabilityRegistry(StrictModel):
    """Aggregated snapshot of all system capabilities."""

    registry_version: str = "0.1"
    entries: list[SystemCapabilityEntry] = []
    generated_at: str = ""
    source_files: list[str] = []
    validation_summary: dict[str, Any] = {}

    @classmethod
    def create(cls, entries: list[SystemCapabilityEntry], source_files: list[str], validation_summary: dict[str, Any]) -> "SystemCapabilityRegistry":
        return cls(
            entries=entries,
            source_files=source_files,
            validation_summary=validation_summary,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )
