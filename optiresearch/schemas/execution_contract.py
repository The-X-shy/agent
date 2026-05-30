"""Execution contract schema for Phase 68."""
from __future__ import annotations

from typing import Any

from optiresearch.memory.schemas import StrictModel


class ExecutionContract(StrictModel):
    """Formal execution contract linking handler, skill, design, and backend."""

    contract_id: str
    handler_id: str
    skill_id: str = ""
    design_ids: list[str] = []
    backend_ids: list[str] = []
    execution_modes: list[str] = []
    input_schema: dict[str, Any] = {}
    output_schema: dict[str, Any] = {}
    required_inputs: list[str] = []
    optional_inputs: list[str] = []
    required_outputs: list[str] = []
    optional_outputs: list[str] = []
    required_metrics: list[str] = []
    optional_metrics: list[str] = []
    status_values: list[str] = []
    evidence_level_mapping: dict[str, str] = {}
    claim_ceiling_mapping: dict[str, str] = {}
    failure_modes: list[str] = []
    retry_policy: dict[str, Any] = {}
    timeout_policy: dict[str, Any] = {}
    artifact_contract_id: str = ""
    report_contract_id: str = ""
