"""Skill contracts for Phase 36 — formalized skill specifications."""

from __future__ import annotations

from typing import Any, Literal, Optional

from optiresearch.memory.schemas import StrictModel


class SkillSpec(StrictModel):
    skill_id: str
    name: str = ""
    description: str = ""
    version: str = "1.0"
    input_schema: dict[str, Any] = {}
    output_schema: dict[str, Any] = {}
    required_backends: list[str] = []
    produced_artifacts: list[str] = []
    evidence_level: Optional[str] = None
    risk_level: Literal["low", "medium", "high"] = "low"
    timeout_sec: int = 600
    allowed_execution_targets: list[str] = ["local", "remote"]
    claim_implications: list[str] = []
    tags: list[str] = []


class SkillResult(StrictModel):
    skill_id: str
    status: Literal["succeeded", "failed", "unsupported"]
    inputs_hash: str = ""
    output: dict[str, Any] = {}
    artifacts: list[str] = []
    events: list[str] = []
    errors: list[str] = []
    warnings: list[str] = []
    execution_time_sec: float = 0.0
