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
            "switch_backend": 8,
            "probe_waveoptics_path": 7,
            "try_alternative_parameterization": 6,
            "redesign_objective": 5,
            "request_real_data": 5,
            "reduce_learning_rate": 4,
            "enable_trust_region": 4,
            "reduce_parameter_dimension": 3,
            "add_accept_tolerance": 3,
            "switch_optimizer": 2,
            "increase_model_capacity": 1,
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
}
