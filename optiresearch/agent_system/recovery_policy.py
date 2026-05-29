"""Recovery policy for Phase 36 — convert failures to actionable recovery plans."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from optiresearch.agent_system.failure_taxonomy import FailureClassifier, FailureMode


class RecoveryPolicy:
    def __init__(self, classifier: FailureClassifier | None = None):
        self._classifier = classifier or FailureClassifier()

    def recommend_recovery(self, failure_id_or_result: str | dict[str, Any],
                           context: dict[str, Any] | None = None) -> dict[str, Any]:
        if isinstance(failure_id_or_result, dict):
            fm = self._classifier.classify(failure_id_or_result)
        else:
            fm = self._classifier.classify_by_id(failure_id_or_result)
        if fm is None:
            return {"error": f"Unknown failure: {failure_id_or_result}"}
        ranked = self._rank_recoveries(fm, context or {})
        return {
            "failure_id": fm.failure_id,
            "category": fm.category,
            "severity": fm.severity,
            "description": fm.description,
            "likely_causes": fm.likely_causes,
            "recoveries": ranked,
            "claim_impact": fm.claim_impact,
        }

    def _rank_recoveries(self, fm: FailureMode, context: dict[str, Any]) -> list[dict[str, Any]]:
        rankings: dict[str, int] = {
            "report_negative_result": 10,
            "report_full_geolens_non_differentiable_path": 10,
            "component_first_fresnel_probe": 9,
            "component_first_binary2phase_probe": 9,
            "diffractive_component_probe": 8,
            "differentiable_surrogate_psf_parameterization": 7,
            "run_native_geolens_optimizer_param_audit": 7,
            "surface_parameter_adapter": 7,
            "switch_backend": 6,
            "probe_waveoptics_path": 6,
            "try_alternative_parameterization": 5,
            "redesign_objective": 4,
            "request_real_data": 4,
            "reduce_learning_rate": 3,
            "enable_trust_region": 3,
            "reduce_parameter_dimension": 2,
            "add_accept_tolerance": 2,
            "switch_optimizer": 1,
            "increase_model_capacity": 1,
            "full_geolens_direct_update": 0,
            "repeated_lr_sweep_on_full_geolens": 0,
        }
        ranked = []
        for rec in fm.recommended_recoveries:
            ranked.append({
                "recovery": rec,
                "priority": rankings.get(rec, 0),
                "explanation": _recovery_explanations.get(rec, ""),
            })
        ranked.sort(key=lambda r: r["priority"], reverse=True)
        return ranked

    def convert_recovery_to_strategy(self, recovery: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        strategy_map = {
            "component_first_fresnel_probe": {
                "strategy_type": "alternative_parameterization",
                "action": "probe Fresnel diffractive component with native DeepLens API",
                "expected_evidence_gain": "medium",
                "risk": "medium",
            },
            "component_first_binary2phase_probe": {
                "strategy_type": "alternative_parameterization",
                "action": "probe Binary2Phase component with polynomial order optimization",
                "expected_evidence_gain": "medium",
                "risk": "medium",
            },
            "diffractive_component_probe": {
                "strategy_type": "alternative_parameterization",
                "action": "probe generalized diffractive component with trainable parameters",
                "expected_evidence_gain": "medium",
                "risk": "medium",
            },
            "differentiable_surrogate_psf_parameterization": {
                "strategy_type": "alternative_parameterization",
                "action": "build differentiable surrogate PSF model as optimization proxy",
                "expected_evidence_gain": "low",
                "risk": "high",
            },
            "surface_parameter_adapter": {
                "strategy_type": "alternative_parameterization",
                "action": "wrap GeoLens surface in nn.Module adapter with autograd",
                "expected_evidence_gain": "low",
                "risk": "high",
            },
            "report_full_geolens_non_differentiable_path": {
                "strategy_type": "report_negative_result",
                "action": "document that full GeoLens geometric path failed the current native optimizer audit",
                "expected_evidence_gain": "low",
                "risk": "low",
            },
            "run_native_geolens_optimizer_param_audit": {
                "strategy_type": "autograd_audit",
                "action": "run GeoLens audit through get_optimizer_params/get_optimizer and float32 geometric PSF",
                "expected_evidence_gain": "high",
                "risk": "low",
            },
            "try_alternative_parameterization": {
                "strategy_type": "alternative_parameterization",
                "action": "run stabilization sweep with DiffractiveLens candidate",
                "expected_evidence_gain": "medium",
                "risk": "medium",
            },
            "redesign_objective": {
                "strategy_type": "objective_redesign",
                "action": "simplify loss function to reduce gradient sensitivity",
                "expected_evidence_gain": "medium",
                "risk": "low",
            },
            "switch_backend": {
                "strategy_type": "backend_switch",
                "action": "switch to diffractive lens or hybrid lens backend",
                "expected_evidence_gain": "medium",
                "risk": "medium",
            },
            "probe_waveoptics_path": {
                "strategy_type": "waveoptics_probe",
                "action": "run full wave-optics coherent ASM probe on WSL",
                "expected_evidence_gain": "high",
                "risk": "high",
            },
            "request_real_data": {
                "strategy_type": "real_data_request",
                "action": "request real HSI measurement data for validation",
                "expected_evidence_gain": "high",
                "risk": "low",
            },
            "report_negative_result": {
                "strategy_type": "report_negative_result",
                "action": "document Phase 35 negative result as structured evidence",
                "expected_evidence_gain": "low",
                "risk": "low",
            },
            "reduce_learning_rate": {
                "strategy_type": "optimizer_change",
                "action": "further reduce optical_lr and add trust region",
                "expected_evidence_gain": "low",
                "risk": "low",
            },
            "reduce_parameter_dimension": {
                "strategy_type": "alternative_parameterization",
                "action": "reduce GeoLens parameter count for simpler optimization",
                "expected_evidence_gain": "medium",
                "risk": "medium",
            },
            "enable_trust_region": {
                "strategy_type": "optimizer_change",
                "action": "enable trust region and PSF stability gating",
                "expected_evidence_gain": "low",
                "risk": "low",
            },
            "switch_optimizer": {
                "strategy_type": "optimizer_change",
                "action": "try SGD with momentum instead of Adam",
                "expected_evidence_gain": "low",
                "risk": "medium",
            },
        }
        return strategy_map.get(recovery, {"strategy_type": "unknown", "action": recovery})

    def explain_recovery(self, recovery: str) -> str:
        return _recovery_explanations.get(recovery, f"No explanation available for: {recovery}")

    def export_recommendation(self, failure_id_or_result: str | dict[str, Any],
                              output_path: str | Path | None = None) -> Path:
        rec = self.recommend_recovery(failure_id_or_result)
        path = Path(output_path or "workspace/reports/recovery_recommendation.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
        return path


_recovery_explanations: dict[str, str] = {
    "component_first_fresnel_probe": "Probe the Fresnel diffractive component via native DeepLens API — test if f0 parameter is trainable via autograd",
    "component_first_binary2phase_probe": "Probe the Binary2Phase component — test polynomial order parameters (order2-order12) with autograd optimization",
    "diffractive_component_probe": "Probe generalized diffractive component candidates with trainable parameters exposed through standard nn.Module",
    "differentiable_surrogate_psf_parameterization": "Build a neural surrogate model of GeoLens PSF that is fully differentiable and can serve as an optimization proxy",
    "surface_parameter_adapter": "Wrap GeoLens surface parameters in a custom nn.Module adapter to expose them through standard parameters() API",
    "run_native_geolens_optimizer_param_audit": "Verify GeoLens trainability through DeepLens get_optimizer_params/get_optimizer and float32 geometric PSF",
    "report_full_geolens_non_differentiable_path": "Document that the current full GeoLens audit failed, with the route remaining conditional on native optimizer API evidence",
    "try_alternative_parameterization": "Switch to a different optical component parameterization (e.g., DiffractiveLens) that may have a smoother optimization landscape",
    "redesign_objective": "Simplify or change the loss function to reduce gradient sensitivity and improve update stability",
    "switch_backend": "Switch to a different optical backend that may provide more stable gradients",
    "probe_waveoptics_path": "Test the full wave-optics (coherent ASM) path which may have different gradient characteristics",
    "request_real_data": "Use real HSI measurements instead of synthetic data to provide a more meaningful optimization target",
    "report_negative_result": "Document the negative result as structured evidence with clear claim boundaries",
    "reduce_learning_rate": "Further reduce the optical learning rate to decrease effective step size",
    "reduce_parameter_dimension": "Use a simpler parameterization with fewer degrees of freedom",
    "enable_trust_region": "Enable trust-region post-step scaling to constrain parameter deltas",
    "add_accept_tolerance": "Allow small loss increases as exploratory updates without rollback",
    "switch_optimizer": "Try a different optimizer that may handle steep landscapes better",
    "full_geolens_direct_update": "Attempt end-to-end GeoLens parameter update only after native optimizer audit confirms connected gradients",
    "repeated_lr_sweep_on_full_geolens": "Re-run LR sweep on full GeoLens — not recommended when autograd graph is disconnected",
}
