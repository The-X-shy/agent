"""Core Pydantic schemas and deterministic ID helpers."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def stable_hash(payload: Any) -> str:
    """Return a deterministic sha256 hash for JSON-like content."""

    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def make_deterministic_id(prefix: str, *parts: Any, length: int = 16) -> str:
    digest = stable_hash(parts)[:length]
    return f"{prefix}_{digest}"


def make_trace_id(workspace_id: str, run_id: str, step_id: Optional[str], actor: str, task: str) -> str:
    return make_deterministic_id("trace", workspace_id, run_id, step_id, actor, task)


def make_artifact_id(
    workspace_id: str,
    run_id: Optional[str],
    trace_id: Optional[str],
    content_hash: str,
    producer: Optional[str] = None,
    name: Optional[str] = None,
) -> str:
    return make_deterministic_id("artifact", workspace_id, run_id, trace_id, content_hash, producer, name)


def make_run_id(workspace_id: str, objective: str, nonce: Optional[str] = None) -> str:
    return make_deterministic_id("run", workspace_id, objective, nonce or datetime.utcnow().isoformat())


def make_claim_id(text: str, scope: Optional[dict[str, Any]] = None) -> str:
    return make_deterministic_id("claim", text, scope or {})


def make_context_pack_id(role: str, intent: str, query: str, scope: Optional[dict[str, Any]] = None) -> str:
    return make_deterministic_id("ctx", role, intent, query, scope or {})


def compute_file_sha256(path: str | Path) -> str:
    """Compute sha256 for a file without loading it all at once."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class MetaTrace(StrictModel):
    trace_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    branch_id: Optional[str]
    step_id: Optional[str]
    actor: Literal[
        "LeadInvestigator",
        "MethodBuilder",
        "SimulationExperimentalist",
        "CriticalReviewer",
        "System",
    ]
    phase: Literal["Explore", "Execute", "Express", "Review"]
    task: str = Field(min_length=1)
    skill_id: Optional[str]
    skill_version: Optional[str]
    tool: Optional[str]
    input_refs: list[str] = Field(default_factory=list)
    output_refs: list[str] = Field(default_factory=list)
    findings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    next_action: Optional[str]
    status: Literal["planned", "running", "succeeded", "failed", "skipped"]
    timestamp_start: Optional[datetime]
    timestamp_end: Optional[datetime]
    parents: list[str] = Field(default_factory=list)
    content_hash: Optional[str]
    metadata: dict[str, Any] = Field(default_factory=dict)


class ArtifactRef(StrictModel):
    artifact_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    run_id: Optional[str]
    trace_id: Optional[str]
    uri: str = Field(min_length=1)
    mime: Optional[str]
    content_hash: str = Field(min_length=64, max_length=64)
    producer: Optional[str]
    metadata: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)


class RunMemory(StrictModel):
    run_id: str = Field(min_length=1)
    version: int = Field(ge=1)
    workspace_id: str = Field(min_length=1)
    objective: str = Field(min_length=1)
    current_status: str = Field(min_length=1)
    best_metrics: dict[str, Any] = Field(default_factory=dict)
    key_decisions: list[str] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)
    source_trace_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DesignRule(StrictModel):
    rule_id: str = Field(min_length=1)
    statement: str = Field(min_length=1)
    scope: list[str] = Field(default_factory=list)
    status: Literal["active", "deprecated", "contradicted", "superseded"]
    confidence: float = Field(ge=0.0, le=1.0)
    supported_by: list[str] = Field(default_factory=list)
    contradicted_by: list[str] = Field(default_factory=list)
    valid_conditions: dict[str, Any] = Field(default_factory=dict)
    invalid_at: Optional[datetime]
    superseded_by: Optional[str]
    source_trace_ids: list[str] = Field(default_factory=list)


class EvidenceEdge(StrictModel):
    artifact_id: str = Field(min_length=1)
    trace_id: Optional[str] = None
    metric_name: Optional[str] = None
    metric_value: Optional[Union[float, str]] = None
    relation: Literal["supports", "contradicts", "qualifies"] = "supports"
    score: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1)
    # Phase 47: artifact metadata for store cross-referencing
    evidence_role: str = ""
    artifact_type: str = ""
    artifact_sha256: str = ""
    remote_job_id: str = ""
    artifact_store_source: str = ""


class ClaimEvidence(StrictModel):
    claim_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    status: Literal[
        "supported",
        "partially_supported",
        "unsupported",
        "contradicted",
        "needs_followup",
        "simulation_only",
        "prototype_validated",
    ]
    support_score: float = Field(ge=0.0, le=1.0)
    support_edges: list[EvidenceEdge] = Field(default_factory=list)
    contradict_edges: list[EvidenceEdge] = Field(default_factory=list)
    scope: dict[str, Any] = Field(default_factory=dict)
    review_status: str = Field(min_length=1)
    required_caveats: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PlanTemplate(StrictModel):
    template_id: str = Field(min_length=1)
    intent: str = Field(min_length=1)
    description: str = Field(min_length=1)
    slots: list[str] = Field(default_factory=list)
    steps: list[str] = Field(default_factory=list)
    preconditions: list[str] = Field(default_factory=list)
    historical_success_rate: float = Field(ge=0.0, le=1.0)
    average_cost: dict[str, Any] = Field(default_factory=dict)
    success_count: int = Field(default=0, ge=0)
    failure_count: int = Field(default=0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SkillMemory(StrictModel):
    skill_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    used_in: list[str] = Field(default_factory=list)
    success_rate: float = Field(ge=0.0, le=1.0)
    preferred_when: list[str] = Field(default_factory=list)
    common_failures: list[str] = Field(default_factory=list)
    best_practices: list[str] = Field(default_factory=list)
    last_updated: Optional[datetime]
    success_count: int = Field(default=0, ge=0)
    failure_count: int = Field(default=0, ge=0)
    commands: list[str] = Field(default_factory=list)
    emitted_artifact_types: list[str] = Field(default_factory=list)


class SkillManifest(StrictModel):
    skill_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    roles: list[str] = Field(default_factory=list)
    intents: list[str] = Field(default_factory=list)
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    permissions: dict[str, Any] = Field(default_factory=dict)
    cost_class: Literal["low", "medium", "high"]
    validators: list[str] = Field(default_factory=list)
    artifacts_emitted: list[str] = Field(default_factory=list)
    evidence_policy: Optional[str]
    sandbox_profile: dict[str, Any] = Field(default_factory=dict)

    @field_validator("skill_id")
    @classmethod
    def skill_id_is_slug(cls, value: str) -> str:
        if not value.replace("-", "").replace("_", "").isalnum():
            raise ValueError("skill_id must be a slug")
        return value
