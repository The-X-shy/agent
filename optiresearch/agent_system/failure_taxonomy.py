"""Failure taxonomy for Phase 36 — structured failure classification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Optional

from optiresearch.memory.schemas import StrictModel


FailureCategory = Literal[
    "backend_unavailable",
    "autograd_break",
    "gradient_instability",
    "claim_overreach",
    "remote_execution_failure",
    "artifact_missing",
    "metric_no_improvement",
    "rollback_all_updates",
    "platform_incompatibility",
    "unsupported_task",
    "unsafe_plan",
]


class FailureMode(StrictModel):
    failure_id: str
    category: FailureCategory
    description: str = ""
    evidence_patterns: dict[str, Any] = {}
    severity: Literal["low", "medium", "high", "critical"] = "medium"
    likely_causes: list[str] = []
    recommended_recoveries: list[str] = []
    claim_impact: str = ""


BUILTIN_FAILURE_MODES: list[FailureMode] = [
    FailureMode(
        failure_id="unstable_native_geolens_update",
        category="gradient_instability",
        description="Native GeoLens geometric PSF optical updates consistently rejected by rollback across all hyperparameter configs",
        evidence_patterns={
            "optical_gradient_norm_min": 1000,
            "accepted_update_count": 0,
            "rollback_count_min": 1,
            "proxy_fallback_used": False,
            "configs_tested_min": 5,
        },
        severity="high",
        likely_causes=[
            "GeoLensCooke geometric parameterization is too sensitive",
            "Optical gradient directions do not align with reconstruction loss improvement",
            "Adam optimizer overshoots in steep parameter landscape",
        ],
        recommended_recoveries=[
            "try_alternative_parameterization",
            "redesign_objective",
            "reduce_parameter_dimension",
            "switch_backend",
            "probe_waveoptics_path",
            "request_real_data",
            "report_negative_result",
        ],
        claim_impact="Native GeoLens optical improvement claim is NOT supported",
    ),
    FailureMode(
        failure_id="backend_unavailable",
        category="backend_unavailable",
        description="Target backend cannot execute experiments",
        evidence_patterns={"status": "unsupported", "error_code_contains": "BACKEND"},
        severity="critical",
        likely_causes=["Platform incompatibility", "API missing", "Dependency not installed"],
        recommended_recoveries=["switch_backend", "use_lightweight_proxy", "install_dependencies"],
        claim_impact="Cannot claim backend-specific evidence",
    ),
    FailureMode(
        failure_id="gradient_instability",
        category="gradient_instability",
        description="Optical gradient norms are too large, causing training instability",
        evidence_patterns={"optical_gradient_norm_min": 100, "accepted_update_count": 0},
        severity="high",
        likely_causes=["High parameter sensitivity", "Poorly conditioned optimization landscape"],
        recommended_recoveries=["reduce_learning_rate", "enable_trust_region", "switch_optimizer", "redesign_objective"],
        claim_impact="Cannot claim stable optical optimization",
    ),
    FailureMode(
        failure_id="claim_overreach",
        category="claim_overreach",
        description="Proposed claim exceeds what evidence supports",
        evidence_patterns={"claim_gate_decision": "downgraded"},
        severity="medium",
        likely_causes=["Insufficient evidence", "Claim too broad for experiment scope"],
        recommended_recoveries=["narrow_claim", "gather_more_evidence", "accept_downgraded_claim"],
        claim_impact="Claim must be downgraded",
    ),
    FailureMode(
        failure_id="rollback_all_updates",
        category="rollback_all_updates",
        description="Every optical update was rejected by rollback protection",
        evidence_patterns={"accepted_update_count": 0, "rejected_update_count_min": 1},
        severity="high",
        likely_causes=["Step size too large", "Loss landscape is steep", "PSF too sensitive to parameters"],
        recommended_recoveries=["reduce_step_size", "enable_trust_region", "add_accept_tolerance", "switch_backend"],
        claim_impact="Cannot claim optical improvement",
    ),
    FailureMode(
        failure_id="remote_execution_failure",
        category="remote_execution_failure",
        description="Remote worker could not execute or return results",
        evidence_patterns={"status": "failed", "error_code_contains": "REMOTE"},
        severity="high",
        likely_causes=["SSH connection issue", "WSL environment mismatch", "Command not allowed"],
        recommended_recoveries=["check_worker_status", "retry_with_backoff", "run_locally"],
        claim_impact="Cannot claim cross-platform validation",
    ),
    FailureMode(
        failure_id="platform_incompatibility",
        category="platform_incompatibility",
        description="Experiment runs on one platform but not another",
        evidence_patterns={"status": "unsupported", "error_code_contains": "INDEXERROR"},
        severity="medium",
        likely_causes=["OS-specific API limitation", "DLL/library mismatch"],
        recommended_recoveries=["use_compatible_platform", "report_platform_limitation", "add_platform_guard"],
        claim_impact="Claim must be scoped to compatible platforms",
    ),
    FailureMode(
        failure_id="metric_no_improvement",
        category="metric_no_improvement",
        description="Experiment completed but produced no metric improvement",
        evidence_patterns={"status": "succeeded", "loss_decreased": False},
        severity="medium",
        likely_causes=["Suboptimal hyperparameters", "Model capacity limit", "Objective misalignment"],
        recommended_recoveries=["hyperparameter_sweep", "redesign_objective", "increase_model_capacity"],
        claim_impact="Cannot claim improvement",
    ),
    # Phase 61: GeoLens autograd diagnostic findings
    FailureMode(
        failure_id="no_standard_trainable_parameters",
        category="autograd_break",
        description="GeoLens does not expose trainable parameters via standard nn.Module.parameters()",
        evidence_patterns={
            "parameter_count": 0,
            "trainable_param_count": 0,
            "status": "succeeded",
        },
        severity="high",
        likely_causes=[
            "GeoLens uses non-standard parameter storage",
            "PSF computation is not wrapped as nn.Module",
            "GeoLens surface parameters are not exposed through parameters()",
        ],
        recommended_recoveries=[
            "component_first_fresnel_probe",
            "component_first_binary2phase_probe",
            "differentiable_surrogate_psf_parameterization",
            "surface_parameter_adapter",
            "report_full_geolens_non_differentiable_path",
        ],
        claim_impact="Direct full GeoLens optical optimization claim is NOT supported",
    ),
    FailureMode(
        failure_id="autograd_graph_disconnected",
        category="autograd_break",
        description="GeoLens PSF output and loss are not connected through autograd graph",
        evidence_patterns={
            "graph_connected": False,
            "psf_requires_grad": False,
            "loss_requires_grad": False,
            "status": "succeeded",
        },
        severity="high",
        likely_causes=[
            "PSF computation uses non-differentiable operations",
            "Detach or no_grad context in GeoLens forward",
            "PSF is computed outside PyTorch autograd",
        ],
        recommended_recoveries=[
            "component_first_fresnel_probe",
            "component_first_binary2phase_probe",
            "diffractive_component_probe",
            "differentiable_surrogate_psf_parameterization",
            "report_full_geolens_non_differentiable_path",
        ],
        claim_impact="Cannot claim differentiable GeoLens optimization",
    ),
    FailureMode(
        failure_id="non_differentiable_geolens_psf_path",
        category="gradient_instability",
        description="GeoLens geometric PSF path is confirmed non-differentiable — optimization requires component-level or surrogate route",
        evidence_patterns={
            "graph_connected": False,
            "parameter_count": 0,
            "status": "succeeded",
        },
        severity="critical",
        likely_causes=[
            "geolens.psf_geometric uses ray tracing, not differentiable rendering",
            "No autograd path from PSF output back to lens parameters",
            "Full GeoLens is designed for analysis, not gradient-based optimization",
        ],
        recommended_recoveries=[
            "component_first_fresnel_probe",
            "component_first_binary2phase_probe",
            "diffractive_component_probe",
            "differentiable_surrogate_psf_parameterization",
            "surface_parameter_adapter",
            "report_full_geolens_non_differentiable_path",
        ],
        claim_impact="Full GeoLens direct update is BLOCKED; component-level or surrogate path required",
    ),
]


class FailureClassifier:
    def classify(self, result_dict: dict[str, Any]) -> Optional[FailureMode]:
        for fm in BUILTIN_FAILURE_MODES:
            if self._matches(fm, result_dict):
                return fm
        return None

    def classify_by_id(self, failure_id: str) -> Optional[FailureMode]:
        for fm in BUILTIN_FAILURE_MODES:
            if fm.failure_id == failure_id:
                return fm
        return None

    def list_all(self) -> list[FailureMode]:
        return list(BUILTIN_FAILURE_MODES)

    def _matches(self, fm: FailureMode, result: dict[str, Any]) -> bool:
        patterns = fm.evidence_patterns
        if not patterns:
            return False
        for key, expected in patterns.items():
            if key.endswith("_min"):
                actual_key = key[:-4]
                if result.get(actual_key, 0) < expected:
                    return False
            elif key.endswith("_contains"):
                actual_key = key[:-9]
                val = str(result.get(actual_key, ""))
                if str(expected).lower() not in val.lower():
                    return False
            else:
                if result.get(key) != expected:
                    return False
        return True

    def export_taxonomy(self, output_path: str | Path | None = None) -> Path:
        path = Path(output_path or "workspace/reports/failure_taxonomy.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [fm.model_dump(mode="json") for fm in BUILTIN_FAILURE_MODES]
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return path
