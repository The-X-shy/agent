"""DeepLens Design Strategy schema for Phase 54."""

from __future__ import annotations
from typing import Any, Literal
from optiresearch.memory.schemas import StrictModel


class DeepLensDesignStrategy(StrictModel):
    strategy_id: str
    name: str = ""
    strategy_family: Literal[
        "curriculum_learning", "optical_regularization", "staged_optimization",
        "component_first", "surface_freeze_unfreeze", "parameterization_reduction",
        "ray_to_wave_progression", "diffractive_probe", "hybrid_probe",
        "negative_result_report",
    ] = "negative_result_report"
    objective: str = ""
    required_backend: str = "deeplens_geolens_geometric"
    execution_target: Literal["local", "remote_opt_in", "dry_run"] = "dry_run"
    evidence_level: str = "diagnostic_evidence"
    claim_ceiling: str = "diagnostic_evidence"
    expected_failure_modes: list[str] = []
    compatible_diagnosis_failure_modes: list[str] = []
    required_skills: list[str] = []
    runtime_cost: str = "low"
    risk_level: str = "low"
    parameters: dict[str, Any] = {}
    caveats: list[str] = []
    enabled: bool = True


class DeepLensDesignStrategyResult(StrictModel):
    strategy_id: str = ""
    status: str = "dry_run"
    selected: bool = False
    generated_design_id: str = ""
    execution_result: dict[str, Any] = {}
    evidence_level: str = ""
    claim_gate_decision: dict[str, Any] = {}
    caveats: list[str] = []
