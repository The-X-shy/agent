"""Evidence-to-Strategy Reasoner for Phase 36.

Converts structured evidence (including negative results) into
candidate research strategies without human prompt engineering.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional


StrategyType = Literal[
    "alternative_parameterization",
    "objective_redesign",
    "backend_switch",
    "waveoptics_probe",
    "real_data_request",
    "report_negative_result",
    "optimizer_change",
    "parameter_reduction",
    "autograd_audit",
    "parameter_inspection",
    "component_inspection",
    "run_probe_only",
    "component_first",
    "surrogate_parameterization",
]


@dataclass
class CandidateStrategy:
    strategy_id: str
    strategy_type: StrategyType
    rationale: str
    expected_evidence_gain: str
    expected_metric_gain: str
    risk: str
    cost: str
    required_skills: list[str] = field(default_factory=list)
    required_backend: str = ""
    proposed_experiment_templates: list[str] = field(default_factory=list)
    claim_ceiling: str = ""
    blocked_route: str = ""
    pivot_reason: str = ""
    required_component_backend: str = ""
    expected_claim_ceiling: str = ""


class EvidenceStrategyReasoner:
    def __init__(self):
        self._strategies: list[CandidateStrategy] = []

    def reason(
        self,
        objective: str = "",
        failure_mode: str = "unstable_native_geolens_update",
        backend_id: str = "deeplens_geolens_geometric",
        context: dict[str, Any] | None = None,
    ) -> list[CandidateStrategy]:
        self._strategies = []

        # Strategy 1: Alternative parameterization
        self._strategies.append(CandidateStrategy(
            strategy_id="alt_param_diffractive",
            strategy_type="alternative_parameterization",
            rationale=(
                "GeoLensCooke geometric parameterization produces optical gradients "
                "with norm >4000, causing all updates to be rejected by rollback. "
                "DiffractiveLens may have a smoother parameter landscape with fewer "
                "degrees of freedom."
            ),
            expected_evidence_gain="medium",
            expected_metric_gain="low",
            risk="medium",
            cost="medium",
            required_skills=["native_geolens_stabilization_sweep"],
            required_backend="deeplens_geolens_geometric",
            proposed_experiment_templates=["stabilization_sweep_diffractive"],
            claim_ceiling="native_lens_simulation",
        ))

        # Strategy 2: Objective redesign
        self._strategies.append(CandidateStrategy(
            strategy_id="objective_redesign_simpler_metric",
            strategy_type="objective_redesign",
            rationale=(
                "Current loss uses MSE + spectral_angle + measurement_consistency, "
                "which may create a complex loss landscape. Simplifying to MSE-only "
                "could produce smoother gradients and allow optical updates to be accepted."
            ),
            expected_evidence_gain="medium",
            expected_metric_gain="medium",
            risk="low",
            cost="low",
            required_skills=["native_geolens_stabilization_sweep"],
            required_backend="deeplens_geolens_geometric",
            proposed_experiment_templates=["stabilization_sweep_simpler_loss"],
            claim_ceiling="native_lens_simulation",
        ))

        # Strategy 3: Backend switch to wave-optics
        self._strategies.append(CandidateStrategy(
            strategy_id="backend_switch_waveoptics",
            strategy_type="waveoptics_probe",
            rationale=(
                "The geometric PSF path uses ray tracing which may produce "
                "non-smooth gradients. The coherent ASM wave-optics path computes "
                "PSF through physical diffraction propagation, potentially producing "
                "better-conditioned gradients."
            ),
            expected_evidence_gain="high",
            expected_metric_gain="medium",
            risk="high",
            cost="high",
            required_skills=["deeplens_waveoptics_probe", "native_waveoptics_hsi_codesign"],
            required_backend="deeplens_coherent_asm",
            proposed_experiment_templates=["waveoptics_probe", "waveoptics_stabilization_sweep"],
            claim_ceiling="native_waveoptics_simulation",
        ))

        # Strategy 4: Report negative result
        self._strategies.append(CandidateStrategy(
            strategy_id="report_negative_result",
            strategy_type="report_negative_result",
            rationale=(
                "Phase 35 tested 30 hyperparameter configurations across "
                "lr=1e-8..1e-6, grad_clip=0.01..1.0, trust_region=1e-4..1e-3 "
                "with PSF stability gating and accept tolerance. "
                "0/30 configs achieved accepted optical updates. "
                "This is a well-characterized negative result that should be "
                "documented as structured evidence."
            ),
            expected_evidence_gain="low",
            expected_metric_gain="low",
            risk="low",
            cost="low",
            required_skills=["report_generation"],
            required_backend="",
            proposed_experiment_templates=["negative_result_report"],
            claim_ceiling="rollback_protected_native_lens_hsi",
        ))

        # Strategy 5: Real data request
        self._strategies.append(CandidateStrategy(
            strategy_id="real_data_request",
            strategy_type="real_data_request",
            rationale=(
                "All experiments to date use synthetic HSI data. Real camera "
                "measurements would provide a more meaningful optimization target, "
                "potentially revealing whether the gradient instability is an artifact "
                "of synthetic data or a fundamental property of the GeoLens parameterization."
            ),
            expected_evidence_gain="high",
            expected_metric_gain="high",
            risk="low",
            cost="medium",
            required_skills=[],
            required_backend="",
            proposed_experiment_templates=["real_data_hsi_codesign"],
            claim_ceiling="real_hsi_validation",
        ))

        # Strategy 6: Parameter reduction
        self._strategies.append(CandidateStrategy(
            strategy_id="param_reduction",
            strategy_type="parameter_reduction",
            rationale=(
                "GeoLensCooke has 13 optical parameters. Reducing to a subset "
                "(e.g., only curvature and thickness) would simplify the optimization "
                "landscape and may allow stable updates."
            ),
            expected_evidence_gain="medium",
            expected_metric_gain="low",
            risk="medium",
            cost="medium",
            required_skills=["native_geolens_stabilization_sweep"],
            required_backend="deeplens_geolens_geometric",
            proposed_experiment_templates=["reduced_param_sweep"],
            claim_ceiling="native_lens_simulation",
        ))

        return self._strategies

    def reason_from_diagnosis(
        self,
        diagnosis: dict[str, Any] | None = None,
        objective: str = "",
    ) -> list[CandidateStrategy]:
        """Generate strategies from a GradientInstabilityDiagnosis."""
        self._strategies = []
        if not diagnosis or diagnosis.get("status") != "diagnosed":
            self._strategies.append(CandidateStrategy(
                strategy_id="insufficient_diagnosis",
                strategy_type="report_negative_result",
                rationale="No valid gradient instability diagnosis available — insufficient evidence for targeted recovery.",
                expected_evidence_gain="low", expected_metric_gain="low",
                risk="low", cost="low", required_skills=["report_generation"],
                claim_ceiling="report_only",
            ))
            return self._strategies

        diag_id = diagnosis.get("diagnosis_id", "unknown")
        failure_modes = diagnosis.get("failure_modes", [])
        likely_causes = diagnosis.get("likely_causes", [])
        recoveries = diagnosis.get("recommended_recoveries", [])
        severity = diagnosis.get("severity", "medium")

        if "no_parameter_change" in failure_modes:
            self._strategies.extend([
                CandidateStrategy(
                    strategy_id="autograd_graph_audit",
                    strategy_type="autograd_audit",
                    rationale="No optical parameter change detected — verify autograd graph integrity and trainable parameter set before attempting further optimization.",
                    expected_evidence_gain="medium", expected_metric_gain="low",
                    risk="low", cost="low", required_skills=["autograd_audit"],
                    claim_ceiling="diagnostic_evidence",
                ),
                CandidateStrategy(
                    strategy_id="verify_trainable_parameters",
                    strategy_type="parameter_inspection",
                    rationale="Verify which GeoLens parameters are trainable and whether gradient flows through the geometric PSF path.",
                    expected_evidence_gain="medium", expected_metric_gain="low",
                    risk="low", cost="low", required_skills=["backend_probe"],
                    claim_ceiling="diagnostic_evidence",
                ),
            ])

        if "unstable_training" in failure_modes or "extreme_gradient_spike" in failure_modes:
            self._strategies.extend([
                CandidateStrategy(
                    strategy_id="objective_redesign_simpler_metric",
                    strategy_type="objective_redesign",
                    rationale="Gradient instability detected — redesign objective to simpler metric (MSE-only) may reduce loss landscape complexity and allow stable updates.",
                    expected_evidence_gain="medium", expected_metric_gain="medium",
                    risk="low", cost="low", required_skills=["lightweight_scientific_hsi_mse_only"],
                    claim_ceiling="lightweight_scientific_execution",
                ),
                CandidateStrategy(
                    strategy_id="param_reduction_lightweight",
                    strategy_type="parameter_reduction",
                    rationale="Excessive gradient norms suggest over-parameterization — reduce trainable optical parameter scope to stabilize optimization.",
                    expected_evidence_gain="medium", expected_metric_gain="low",
                    risk="low", cost="low", required_skills=["param_reduction_sweep"],
                    claim_ceiling="lightweight_scientific_execution",
                ),
            ])

        if "loss_not_improved" in failure_modes:
            self._strategies.append(CandidateStrategy(
                strategy_id="objective_with_psf_regularization",
                strategy_type="objective_redesign",
                rationale="Loss not improved — add PSF regularization to prevent degenerate optical solutions.",
                expected_evidence_gain="medium", expected_metric_gain="medium",
                risk="low", cost="low", required_skills=["lightweight_scientific_hsi_mse_only"],
                claim_ceiling="lightweight_scientific_execution",
            ))

        if "gradient_flow_blocked" in likely_causes:
            self._strategies.append(CandidateStrategy(
                strategy_id="component_level_geolens_probe",
                strategy_type="component_inspection",
                rationale="Gradient flow appears blocked — probe individual GeoLens components to identify the breakpoint.",
                expected_evidence_gain="high", expected_metric_gain="low",
                risk="medium", cost="medium", required_skills=["backend_probe"],
                claim_ceiling="diagnostic_evidence",
            ))

        # Phase 61: GeoLens autograd diagnostic findings — pivot to component-level
        if "no_standard_trainable_parameters" in failure_modes:
            self._strategies.append(CandidateStrategy(
                strategy_id="component_first_fresnel_probe",
                strategy_type="component_first",
                rationale="GeoLens has no standard trainable parameters — pivot to Fresnel diffractive component which exposes f0 as trainable via nn.Module.",
                expected_evidence_gain="medium", expected_metric_gain="low",
                risk="medium", cost="medium", required_skills=["deeplens_component_first_probe"],
                required_backend="deeplens_fresnel_component",
                claim_ceiling="diagnostic_evidence",
                blocked_route="full_geolens_direct_update",
                pivot_reason="GeoLens does not expose trainable parameters through standard nn.Module.parameters()",
                required_component_backend="deeplens_fresnel_component",
                expected_claim_ceiling="native_component_optimization",
            ))

        if "autograd_graph_disconnected" in failure_modes:
            self._strategies.extend([
                CandidateStrategy(
                    strategy_id="component_first_binary2phase_probe",
                    strategy_type="component_first",
                    rationale="GeoLens autograd graph is disconnected — pivot to Binary2Phase component with polynomial order optimization (order2-order12).",
                    expected_evidence_gain="medium", expected_metric_gain="low",
                    risk="medium", cost="medium", required_skills=["deeplens_component_first_probe"],
                    required_backend="deeplens_binary2phase_component",
                    claim_ceiling="diagnostic_evidence",
                    blocked_route="full_geolens_direct_update",
                    pivot_reason="GeoLens PSF and loss are not connected through autograd graph",
                    required_component_backend="deeplens_binary2phase_component",
                    expected_claim_ceiling="native_component_optimization",
                ),
                CandidateStrategy(
                    strategy_id="differentiable_surrogate_psf",
                    strategy_type="surrogate_parameterization",
                    rationale="GeoLens autograd is disconnected — consider building a differentiable neural surrogate of the PSF.",
                    expected_evidence_gain="low", expected_metric_gain="low",
                    risk="high", cost="high", required_skills=["backend_probe"],
                    claim_ceiling="diagnostic_evidence",
                    blocked_route="full_geolens_direct_update",
                    pivot_reason="Surrogate model needed when native autograd is unavailable",
                ),
            ])

        probes_succeeded = set()
        for recovery in recoveries:
            if isinstance(recovery, str) and recovery.startswith("component_probe_succeeded:"):
                probes_succeeded.add(recovery.split(":", 1)[1])
        geolens_blocked = (
            "full_geolens_direct_update_blocked" in failure_modes
            or any(s.blocked_route == "full_geolens_direct_update" for s in self._strategies)
        )
        if geolens_blocked and ("fresnel" in probes_succeeded or "binary2phase" in probes_succeeded):
            if "fresnel" in probes_succeeded:
                self._strategies.append(CandidateStrategy(
                    strategy_id="component_surrogate_hsi_codesign_fresnel",
                    strategy_type="surrogate_parameterization",
                    rationale="Full GeoLens direct update is blocked, and Fresnel component probing succeeded; wire the component into a differentiable surrogate PSF HSI loop.",
                    expected_evidence_gain="high",
                    expected_metric_gain="medium",
                    risk="low",
                    cost="low",
                    required_skills=["component_surrogate_hsi_codesign"],
                    required_backend="component_surrogate_psf",
                    proposed_experiment_templates=["component_surrogate_fresnel_hsi_codesign_design"],
                    claim_ceiling="component_surrogate_hsi_codesign",
                    blocked_route="full_geolens_direct_update",
                    pivot_reason="Component probe succeeded while full GeoLens direct update remains blocked",
                    required_component_backend="deeplens_fresnel_component",
                    expected_claim_ceiling="component_surrogate_hsi_codesign",
                ))
            if "binary2phase" in probes_succeeded:
                self._strategies.append(CandidateStrategy(
                    strategy_id="component_surrogate_hsi_codesign_binary2phase",
                    strategy_type="surrogate_parameterization",
                    rationale="Full GeoLens direct update is blocked, and Binary2Phase component probing succeeded; wire polynomial phase parameters into a differentiable surrogate PSF HSI loop.",
                    expected_evidence_gain="high",
                    expected_metric_gain="medium",
                    risk="low",
                    cost="low",
                    required_skills=["component_surrogate_hsi_codesign"],
                    required_backend="component_surrogate_psf",
                    proposed_experiment_templates=["component_surrogate_binary2phase_hsi_codesign_design"],
                    claim_ceiling="component_surrogate_hsi_codesign",
                    blocked_route="full_geolens_direct_update",
                    pivot_reason="Component probe succeeded while full GeoLens direct update remains blocked",
                    required_component_backend="deeplens_binary2phase_component",
                    expected_claim_ceiling="component_surrogate_hsi_codesign",
                ))

        if "non_differentiable_geolens_psf_path" in failure_modes:
            self._strategies.extend([
                CandidateStrategy(
                    strategy_id="component_first_fresnel_probe",
                    strategy_type="component_first",
                    rationale="Full GeoLens PSF path is confirmed non-differentiable — probe Fresnel component as alternative.",
                    expected_evidence_gain="medium", expected_metric_gain="low",
                    risk="medium", cost="medium", required_skills=["deeplens_component_first_probe"],
                    required_backend="deeplens_fresnel_component",
                    claim_ceiling="diagnostic_evidence",
                    blocked_route="full_geolens_direct_update",
                    pivot_reason="geolens.psf_geometric uses ray tracing, not differentiable rendering",
                    required_component_backend="deeplens_fresnel_component",
                    expected_claim_ceiling="native_component_optimization",
                ),
                CandidateStrategy(
                    strategy_id="surface_parameter_adapter",
                    strategy_type="surrogate_parameterization",
                    rationale="Wrap GeoLens surface parameters in custom nn.Module adapter for autograd compatibility.",
                    expected_evidence_gain="low", expected_metric_gain="low",
                    risk="high", cost="high", required_skills=["backend_probe"],
                    claim_ceiling="diagnostic_evidence",
                    blocked_route="full_geolens_direct_update",
                    pivot_reason="Adapter needed to bridge GeoLens parameters into autograd",
                ),
            ])

        # Always include report option
        if severity in ("high", "critical"):
            self._strategies.append(CandidateStrategy(
                strategy_id="record_negative_result_diagnosis",
                strategy_type="report_negative_result",
                rationale=f"Diagnosis severity={severity} — record as structured negative result with diagnosis {diag_id}.",
                expected_evidence_gain="low", expected_metric_gain="low",
                risk="low", cost="low", required_skills=["report_generation"],
                claim_ceiling="report_only",
            ))

        # Attach diagnosis metadata
        for s in self._strategies:
            if not hasattr(s, 'diagnosis_id') or not s.diagnosis_id:
                pass  # CandidateStrategy is a dataclass without these fields

        # Phase 54: Add DeepLens design strategies from registry
        try:
            from optiresearch.optics.deeplens_design_strategy_registry import (
                get_deeplens_design_strategy_registry,
            )
            dl_registry = get_deeplens_design_strategy_registry()
            dl_strategies = dl_registry.find_for_diagnosis(failure_modes)
            for ds in dl_strategies:
                evidence_gain = "medium" if ds.strategy_family != "negative_result_report" else "low"
                self._strategies.append(CandidateStrategy(
                    strategy_id=ds.strategy_id,
                    strategy_type=ds.strategy_family,
                    rationale=ds.objective,
                    expected_evidence_gain=evidence_gain,
                    expected_metric_gain="low",
                    risk=ds.risk_level, cost=ds.runtime_cost,
                    required_skills=ds.required_skills,
                    claim_ceiling=ds.claim_ceiling,
                ))
        except Exception:
            pass

        if not self._strategies:
            self._strategies.append(CandidateStrategy(
                strategy_id="run_probe_only_diagnosis",
                strategy_type="run_probe_only",
                rationale="Diagnosis could not determine specific recovery — run lightweight probe to gather more data.",
                expected_evidence_gain="low", expected_metric_gain="low",
                risk="low", cost="low", required_skills=["backend_probe"],
                claim_ceiling="diagnostic_evidence",
            ))

        return self._strategies

    def get_top_strategy(self) -> Optional[CandidateStrategy]:
        if not self._strategies:
            return None
        priority_order = [
            "report_negative_result",
            "component_first",
            "alternative_parameterization",
            "surrogate_parameterization",
            "objective_redesign",
            "backend_switch",
            "waveoptics_probe",
            "parameter_reduction",
            "component_inspection",
            "autograd_audit",
            "parameter_inspection",
            "real_data_request",
            "optimizer_change",
        ]
        ranked = sorted(
            self._strategies,
            key=lambda s: priority_order.index(s.strategy_type) if s.strategy_type in priority_order else 99
        )
        return ranked[0]

    def export(self, output_path: str | Path | None = None) -> Path:
        path = Path(output_path or "workspace/reports/evidence_strategy_recommendations.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [
            {
                "strategy_id": s.strategy_id,
                "strategy_type": s.strategy_type,
                "rationale": s.rationale,
                "expected_evidence_gain": s.expected_evidence_gain,
                "expected_metric_gain": s.expected_metric_gain,
                "risk": s.risk,
                "cost": s.cost,
                "required_skills": s.required_skills,
                "required_backend": s.required_backend,
                "claim_ceiling": s.claim_ceiling,
            }
            for s in self._strategies
        ]
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def export_markdown(self, output_path: str | Path | None = None) -> Path:
        path = Path(output_path or "workspace/reports/evidence_strategy_recommendations.md")
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Evidence-to-Strategy Recommendations",
            "",
            "| # | Strategy | Type | Evidence Gain | Risk | Cost | Rationale |",
            "|---|---|---|---|---|---|---|",
        ]
        for i, s in enumerate(self._strategies, 1):
            lines.append(
                f"| {i} | {s.strategy_id} | {s.strategy_type} | "
                f"{s.expected_evidence_gain} | {s.risk} | {s.cost} | "
                f"{s.rationale[:80]}... |"
            )
        top = self.get_top_strategy()
        if top:
            lines.extend(["", "## Top Strategy", f"**{top.strategy_id}**: {top.rationale}"])
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path
