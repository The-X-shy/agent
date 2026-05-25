"""Gradient Instability Diagnosis schema for Phase 51."""

from __future__ import annotations

from typing import Any, Literal, Optional

from optiresearch.memory.schemas import StrictModel


class GradientInstabilityMetrics(StrictModel):
    optical_gradient_norm_max: Optional[float] = None
    optical_gradient_norm_mean: Optional[float] = None
    recon_gradient_norm: Optional[float] = None
    accepted_update_count: int = 0
    rejected_update_count: int = 0
    rollback_count: int = 0
    rollback_rate: float = 0.0
    reconstruction_loss_before: Optional[float] = None
    reconstruction_loss_after: Optional[float] = None
    loss_delta: Optional[float] = None
    stable_training_succeeded: bool = False
    optical_parameters_changed: bool = False
    psf_energy_delta: Optional[float] = None
    psf_centroid_delta: Optional[float] = None
    psf_width_delta: Optional[float] = None
    evidence_level: str = ""
    execution_fidelity: str = ""
    proxy_fallback_used: bool = False


class GradientInstabilityDiagnosis(StrictModel):
    diagnosis_id: str = ""
    status: Literal["diagnosed", "insufficient_evidence", "failed"] = "insufficient_evidence"
    source_paths: list[str] = []
    source_count: int = 0
    metrics: GradientInstabilityMetrics = GradientInstabilityMetrics()
    failure_modes: list[str] = []
    severity: Literal["low", "medium", "high", "critical"] = "medium"
    likely_causes: list[str] = []
    recommended_recoveries: list[str] = []
    claim_implications: list[str] = []
    next_experiment_design_hints: list[str] = []
    caveats: list[str] = []
    warnings: list[str] = []
    metadata: dict[str, Any] = {}
