"""Unified agent event model for Phase 36."""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal, Optional

from optiresearch.memory.schemas import StrictModel


EventType = Literal[
    "experiment_requested",
    "experiment_started",
    "experiment_completed",
    "experiment_failed",
    "evidence_added",
    "claim_checked",
    "claim_downgraded",
    "strategy_recommended",
    "memory_updated",
    "skill_called",
    "skill_failed",
    "backend_switched",
    "rollback_triggered",
    "negative_result_recorded",
    "recovery_recommended",
    "state_snapshot_saved",
    "self_test_completed",
    "remote_execution_requested",
    "remote_execution_started",
    "remote_execution_completed",
    "remote_execution_failed",
    "remote_validation_passed",
    "remote_validation_failed",
    "artifact_ingested",
]

SourceModule = Literal[
    "planner",
    "controller",
    "backend",
    "memory",
    "claim_gate",
    "strategy_engine",
    "skill_runtime",
    "remote_executor",
    "reporter",
    "self_test",
    "benchmark",
]

Severity = Literal["info", "warning", "error", "critical"]


class AgentEvent(StrictModel):
    event_id: str = ""
    timestamp: float = 0.0
    event_type: EventType
    source_module: SourceModule
    payload: dict[str, Any] = {}
    severity: Severity = "info"
    related_run_id: Optional[str] = None
    related_job_id: Optional[str] = None
    related_claim_id: Optional[str] = None
    related_artifact_ids: list[str] = []

    @classmethod
    def create(
        cls,
        event_type: EventType,
        source_module: SourceModule,
        payload: dict[str, Any] | None = None,
        severity: Severity = "info",
        related_run_id: Optional[str] = None,
        related_job_id: Optional[str] = None,
        related_claim_id: Optional[str] = None,
        related_artifact_ids: Optional[list[str]] = None,
    ) -> "AgentEvent":
        return cls(
            event_id=str(uuid.uuid4())[:12],
            timestamp=time.time(),
            event_type=event_type,
            source_module=source_module,
            payload=payload or {},
            severity=severity,
            related_run_id=related_run_id,
            related_job_id=related_job_id,
            related_claim_id=related_claim_id,
            related_artifact_ids=related_artifact_ids or [],
        )
