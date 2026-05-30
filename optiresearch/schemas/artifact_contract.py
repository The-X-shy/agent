"""Artifact contract schema for Phase 68."""
from __future__ import annotations

from typing import Literal

from optiresearch.memory.schemas import StrictModel


class ArtifactContract(StrictModel):
    """Contract specifying required and optional artifacts for an output type."""

    contract_id: str
    handler_id: str = ""
    required_artifacts: list[str] = []
    optional_artifacts: list[str] = []
    artifact_roles: dict[str, str] = {}
    sha256_required: bool = True
    artifactstore_registration_required: bool = True
    evidence_binding_required: bool = False
    missing_artifact_policy: Literal["needs_followup", "partial_evidence", "structured_warning"] = "structured_warning"
