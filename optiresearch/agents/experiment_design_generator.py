"""Experiment Design Generator for Phase 36.

Converts CandidateStrategy objects into concrete ExperimentSpecV2 designs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from optiresearch.agents.evidence_strategy_reasoner import CandidateStrategy


@dataclass
class ExperimentDesignCandidate:
    design_id: str
    objective: str
    backend_id: str
    task_type: str
    spec_payload: dict[str, Any] = field(default_factory=dict)
    expected_evidence_level: str = ""
    expected_failure_modes: list[str] = field(default_factory=list)
    # Phase 54-55: DeepLens strategy integration
    source_strategy_id: str = ""
    deeplens_strategy_id: str = ""
    strategy_family: str = ""
    design_origin: str = ""  # diagnosis_reasoner, deeplens_strategy_registry, evidence_strategy_reasoner, manual, fallback
    required_backend_capability: list[str] = field(default_factory=list)
    handler_capability_id: str = ""
    expected_handler_id: str = ""
    expected_skill_id: str = ""
    design_caveats: list[str] = field(default_factory=list)
    claim_boundary_notes: list[str] = field(default_factory=list)
    probe_only: bool = False
    scientific_success_claim_allowed: bool = False
    required_skills: list[str] = field(default_factory=list)
    claim_ceiling: str = ""
    estimated_runtime_sec: int = 600
    risk_level: str = "low"
    handler_id: str = ""
    actual_handler_evidence_level: str = ""
    evidence_alignment_status: str = ""  # aligned, downgraded_to_handler_capability, unsupported
    evidence_downgrade_reason: str = ""


class ExperimentDesignGenerator:
    def generate_designs(
        self,
        strategies: list[CandidateStrategy],
        constraints: dict[str, Any] | None = None,
    ) -> list[ExperimentDesignCandidate]:
        designs: list[ExperimentDesignCandidate] = []
        # Phase 55: Dedicated DeepLens strategy mappings
        dl_designs = self.generate_deeplens_strategy_designs(strategies)
        designs.extend(dl_designs)
        for s in strategies:
            d = self._generate_for_strategy(s)
            if d:
                designs.extend(d)
        return self._align_evidence_levels(designs)

    def generate_deeplens_strategy_designs(
        self,
        strategies: list[CandidateStrategy],
    ) -> list[ExperimentDesignCandidate]:
        """Generate dedicated ExperimentDesignCandidates for DeepLens strategies."""
        designs: list[ExperimentDesignCandidate] = []
        dl_families = {
            "curriculum_learning", "optical_regularization", "staged_optimization",
            "component_first", "surface_freeze_unfreeze", "parameterization_reduction",
            "ray_to_wave_progression", "diffractive_probe", "hybrid_probe",
            "negative_result_report",
        }
        for s in strategies:
            if s.strategy_type not in dl_families:
                continue
            d = self._map_deeplens_strategy(s)
            if d:
                designs.append(d)
        return designs

    def _map_deeplens_strategy(self, s: CandidateStrategy) -> ExperimentDesignCandidate | None:
        sid = s.strategy_id
        base = dict(
            source_strategy_id=sid, deeplens_strategy_id=sid,
            strategy_family=s.strategy_type,
            design_origin="deeplens_strategy_registry",
            risk_level=s.risk, required_skills=s.required_skills,
            claim_ceiling=s.claim_ceiling, probe_only=True,
            scientific_success_claim_allowed=False,
        )
        if sid == "geolens_curriculum_probe":
            return ExperimentDesignCandidate(
                design_id="deeplens_curriculum_probe_design",
                objective="Run staged curriculum probe to avoid direct full-parameter update",
                backend_id="deeplens_geolens_geometric", task_type="curriculum_probe",
                expected_evidence_level="diagnostic_evidence",
                expected_skill_id="deeplens_curriculum_probe",
                expected_handler_id="deeplens_curriculum_probe",
                estimated_runtime_sec=120, **base,
                claim_boundary_notes=["Curriculum probe only — not a validated optical design improvement"],
            )
        if sid == "geolens_regularized_probe":
            return ExperimentDesignCandidate(
                design_id="deeplens_regularized_probe_design",
                objective="Add PSF regularization to dampen gradient spikes",
                backend_id="deeplens_geolens_geometric", task_type="regularized_probe",
                expected_evidence_level="diagnostic_evidence",
                expected_skill_id="deeplens_regularized_probe",
                expected_handler_id="deeplens_regularized_probe",
                estimated_runtime_sec=120, **base,
                claim_boundary_notes=["Regularization probe — not a validated optical design improvement"],
            )
        if sid == "geolens_surface_freeze_unfreeze_probe":
            return ExperimentDesignCandidate(
                design_id="deeplens_surface_freeze_unfreeze_probe_design",
                objective="Inspect trainable surface subsets and freeze unstable surfaces",
                backend_id="deeplens_geolens_geometric", task_type="trainable_parameter_inspection",
                expected_evidence_level="diagnostic_evidence",
                expected_skill_id="deeplens_trainable_parameter_inspection",
                expected_handler_id="deeplens_trainable_parameter_inspection",
                estimated_runtime_sec=300, **base,
                claim_boundary_notes=["Surface inspection — does not confirm optical design improvement"],
            )
        if sid == "component_first_fresnel_probe":
            return ExperimentDesignCandidate(
                design_id="deeplens_component_first_fresnel_probe_design",
                objective="Verify stable differentiable Fresnel component update",
                backend_id="deeplens_fresnel_component", task_type="component_first_probe",
                expected_evidence_level="diagnostic_evidence",
                expected_skill_id="deeplens_component_first_probe",
                expected_handler_id="deeplens_component_first_probe",
                spec_payload={"component": "fresnel"},
                estimated_runtime_sec=300, **base,
                claim_boundary_notes=["Component probe — not lens-level optical design validation"],
            )
        if sid == "component_first_binary2phase_probe":
            return ExperimentDesignCandidate(
                design_id="deeplens_component_first_binary2phase_probe_design",
                objective="Verify differentiable binary2phase component update",
                backend_id="deeplens_binary2phase_component", task_type="component_first_probe",
                expected_evidence_level="diagnostic_evidence",
                expected_skill_id="deeplens_component_first_probe",
                expected_handler_id="deeplens_component_first_probe",
                spec_payload={"component": "binary2phase"},
                estimated_runtime_sec=300, **base,
            )
        if sid == "ray_to_wave_progression_probe":
            return ExperimentDesignCandidate(
                design_id="deeplens_ray_to_wave_progression_probe_design",
                objective="Probe geometric path first; coherent wave only if autograd passes",
                backend_id="deeplens_geolens_geometric", task_type="ray_to_wave_progression_probe",
                expected_evidence_level="diagnostic_evidence",
                expected_skill_id="backend_probe",
                estimated_runtime_sec=600, **base,
                design_caveats=["Full wave-optics claim requires autograd verification — not yet supported"],
            )
        if sid == "diffractive_candidate_probe":
            return ExperimentDesignCandidate(
                design_id="deeplens_diffractive_candidate_probe_design",
                objective="Check diffractive candidate availability and differentiability",
                backend_id="deeplens_diffractive_candidate", task_type="diffractive_candidate_probe",
                expected_evidence_level="diagnostic_evidence",
                expected_skill_id="deeplens_component_first_probe",
                expected_handler_id="deeplens_component_first_probe",
                spec_payload={"component": "diffractive_candidate"},
                estimated_runtime_sec=300, **base,
            )
        if sid == "report_geolens_negative_result":
            return ExperimentDesignCandidate(
                design_id="deeplens_geolens_negative_result_report_design",
                objective="Document GeoLensCooke full parameterization gradient instability",
                backend_id="", task_type="negative_result_report",
                expected_evidence_level="report_only",
                expected_skill_id="report_generation",
                expected_handler_id="report_negative_result_doc",
                estimated_runtime_sec=60, probe_only=False,
                source_strategy_id=sid, deeplens_strategy_id=sid,
                strategy_family=s.strategy_type,
                design_origin="deeplens_strategy_registry",
                risk_level=s.risk, required_skills=s.required_skills,
                claim_ceiling="report_only",
                claim_boundary_notes=["Report-only — does not support optical improvement claims"],
            )
        return None

    def _align_evidence_levels(
        self, designs: list[ExperimentDesignCandidate]
    ) -> list[ExperimentDesignCandidate]:
        from optiresearch.skills.handler_capability_registry import (
            get_handler_capability_registry,
        )
        registry = get_handler_capability_registry()
        for d in designs:
            cap = registry.find_by_design_id(d.design_id)
            if cap is None:
                d.evidence_alignment_status = "unsupported"
                d.evidence_downgrade_reason = f"No handler capability registered for {d.design_id}"
                continue
            d.handler_id = cap.handler_id
            d.actual_handler_evidence_level = cap.actual_evidence_level
            if d.expected_evidence_level == cap.actual_evidence_level:
                d.evidence_alignment_status = "aligned"
            elif _evidence_rank(d.expected_evidence_level) > _evidence_rank(cap.actual_evidence_level):
                d.evidence_alignment_status = "downgraded_to_handler_capability"
                d.evidence_downgrade_reason = (
                    f"Strategy target {d.expected_evidence_level} downgraded to "
                    f"handler capability {cap.actual_evidence_level}"
                )
                d.expected_evidence_level = cap.actual_evidence_level
                d.claim_ceiling = cap.max_claim_ceiling
            else:
                d.evidence_alignment_status = "aligned"
        return designs

    def _generate_for_strategy(self, s: CandidateStrategy) -> list[ExperimentDesignCandidate]:
        if s.strategy_type == "alternative_parameterization":
            return [ExperimentDesignCandidate(
                design_id=f"{s.strategy_id}_sweep",
                objective=f"Test {s.strategy_id} with stabilization sweep",
                backend_id=s.required_backend or "deeplens_geolens_geometric",
                task_type="stable_lens_hsi_codesign",
                spec_payload={
                    "candidate": "DiffractiveLens",
                    "reconstructor": "differentiable_linear",
                    "max_steps": 5,
                    "optical_lr": 1e-7,
                    "optical_grad_clip": 0.1,
                    "trust_region_enabled": True,
                    "max_optical_param_delta": 1e-4,
                    "rollback_on_psf_instability": True,
                    "accept_tolerance": 1e-6,
                },
                expected_evidence_level="native_lens_simulation",
                expected_failure_modes=["gradient_instability", "rollback_all_updates"],
                required_skills=s.required_skills,
                claim_ceiling=s.claim_ceiling,
                estimated_runtime_sec=3600,
                risk_level=s.risk,
            )]
        elif s.strategy_type == "objective_redesign":
            return [ExperimentDesignCandidate(
                design_id=f"{s.strategy_id}_mse_only",
                objective="Test MSE-only loss for smoother optimization",
                backend_id=s.required_backend or "deeplens_geolens_geometric",
                task_type="stable_lens_hsi_codesign",
                spec_payload={
                    "candidate": "GeoLensCooke",
                    "reconstructor": "differentiable_linear",
                    "max_steps": 5,
                    "optical_lr": 1e-7,
                    "optical_grad_clip": 0.1,
                    "loss_weights": {"mse": 1.0, "spectral_angle": 0.0, "measurement_consistency": 0.0},
                    "trust_region_enabled": True,
                    "max_optical_param_delta": 1e-4,
                    "accept_tolerance": 1e-6,
                },
                expected_evidence_level="native_lens_simulation",
                expected_failure_modes=["gradient_instability"],
                required_skills=s.required_skills,
                claim_ceiling=s.claim_ceiling,
                estimated_runtime_sec=3600,
                risk_level=s.risk,
            )]
        elif s.strategy_type == "waveoptics_probe":
            return [ExperimentDesignCandidate(
                design_id=f"{s.strategy_id}_coherent",
                objective="Probe full wave-optics coherent ASM path on WSL",
                backend_id="deeplens_coherent_asm",
                task_type="native_waveoptics_codesign",
                spec_payload={
                    "candidate": "GeoLensCooke",
                    "reconstructor": "differentiable_linear",
                    "max_steps": 3,
                    "optical_lr": 1e-7,
                    "full_wave_optics": True,
                    "phase_to_fft_proxy_used": False,
                },
                expected_evidence_level="native_waveoptics_simulation",
                expected_failure_modes=["backend_unavailable", "autograd_break"],
                required_skills=s.required_skills,
                claim_ceiling=s.claim_ceiling,
                estimated_runtime_sec=3600,
                risk_level=s.risk,
            )]
        elif s.strategy_type == "report_negative_result":
            return [ExperimentDesignCandidate(
                design_id=f"{s.strategy_id}_doc",
                objective="Document Phase 35 negative result as structured evidence",
                backend_id="",
                task_type="native_lens_simulation_codesign",
                spec_payload={"action": "export_system_subunit_report"},
                expected_evidence_level="negative_result",
                expected_failure_modes=[],
                required_skills=["report_generation"],
                claim_ceiling=s.claim_ceiling,
                estimated_runtime_sec=60,
                risk_level="low",
            )]
        elif s.strategy_type == "autograd_audit":
            return [ExperimentDesignCandidate(
                design_id=f"{s.strategy_id}_design",
                objective="Audit autograd graph for gradient breakpoints",
                backend_id="deeplens_geolens_geometric",
                task_type="autograd_audit",
                spec_payload={"diagnostic": "autograd_graph"},
                expected_evidence_level="diagnostic_evidence",
                expected_failure_modes=["autograd_break", "no_gradient"],
                required_skills=["autograd_audit"],
                claim_ceiling="diagnostic_evidence",
                estimated_runtime_sec=300,
                risk_level=s.risk,
            )]
        elif s.strategy_type == "parameter_inspection":
            return [ExperimentDesignCandidate(
                design_id=f"{s.strategy_id}_design",
                objective="Probe trainable parameters and verify gradient flow",
                backend_id="deeplens_geolens_geometric",
                task_type="backend_probe",
                spec_payload={"probe_type": "trainable_parameters"},
                expected_evidence_level="diagnostic_evidence",
                expected_failure_modes=["gradient_flow_blocked"],
                required_skills=["backend_probe"],
                claim_ceiling="diagnostic_evidence",
                estimated_runtime_sec=300,
                risk_level=s.risk,
            )]
        elif s.strategy_type == "component_inspection":
            return [ExperimentDesignCandidate(
                design_id=f"{s.strategy_id}_design",
                objective="Probe individual GeoLens components for gradient breakpoints",
                backend_id="deeplens_geolens_geometric",
                task_type="backend_probe",
                spec_payload={"probe_type": "component_level", "depth": "deep"},
                expected_evidence_level="diagnostic_evidence",
                expected_failure_modes=["deepens_unavailable", "geolens_api_error"],
                required_skills=["backend_probe"],
                claim_ceiling="diagnostic_evidence",
                estimated_runtime_sec=600,
                risk_level=s.risk,
            )]
        elif s.strategy_type == "real_data_request":
            return [ExperimentDesignCandidate(
                design_id=f"{s.strategy_id}_req",
                objective="Request real HSI measurement data for validation",
                backend_id="",
                task_type="native_lens_simulation_codesign",
                spec_payload={"action": "request_real_data"},
                expected_evidence_level="real_hsi",
                expected_failure_modes=["data_unavailable"],
                required_skills=[],
                claim_ceiling=s.claim_ceiling,
                estimated_runtime_sec=0,
                risk_level="low",
            )]
        elif s.strategy_type == "parameter_reduction":
            return [ExperimentDesignCandidate(
                design_id=f"{s.strategy_id}_sweep",
                objective="Test reduced parameterization with stabilization sweep",
                backend_id=s.required_backend or "deeplens_geolens_geometric",
                task_type="stable_lens_hsi_codesign",
                spec_payload={
                    "candidate": "GeoLensCooke",
                    "reconstructor": "differentiable_linear",
                    "max_steps": 5,
                    "optical_lr": 1e-7,
                    "optical_grad_clip": 0.1,
                    "trust_region_enabled": True,
                    "max_optical_param_delta": 1e-4,
                    "param_subset": ["curvature", "thickness"],
                },
                expected_evidence_level="native_lens_simulation",
                expected_failure_modes=["gradient_instability"],
                required_skills=s.required_skills,
                claim_ceiling=s.claim_ceiling,
                estimated_runtime_sec=3600,
                risk_level=s.risk,
            )]
        elif s.strategy_type == "optimizer_change" and s.strategy_id == "full_geolens_geometric_direct_update":
            return [ExperimentDesignCandidate(
                design_id="full_geolens_geometric_direct_update_design",
                objective="Run a guarded full GeoLens geometric PSF native simulation update",
                backend_id="deeplens_geolens_geometric",
                task_type="stable_lens_hsi_codesign",
                spec_payload={
                    "candidate": "GeoLensCooke",
                    "dataset": "synthetic",
                    "reconstructor": "differentiable_linear",
                    "max_steps": 3,
                    "optical_lr": 1e-6,
                    "optical_grad_clip": 1.0,
                    "trust_region_enabled": True,
                    "max_optical_param_delta": 1e-4,
                    "rollback_on_loss_increase": True,
                    "device": "cpu",
                },
                expected_evidence_level="native_lens_simulation",
                expected_failure_modes=["gradient_instability", "rollback_all_updates"],
                source_strategy_id=s.strategy_id,
                strategy_family=s.strategy_type,
                design_origin="evidence_strategy_reasoner",
                required_skills=s.required_skills,
                claim_ceiling="native_lens_simulation",
                estimated_runtime_sec=600,
                risk_level=s.risk,
                expected_handler_id="deeplens_native_geolens_hsi_codesign",
                handler_capability_id="deeplens_native_geolens_hsi_codesign",
                claim_boundary_notes=[
                    "GeoLens geometric PSF native simulation only — not coherent wave-optics",
                    "Synthetic HSI only — no real HSI performance claim",
                ],
            )]
        elif s.strategy_type == "surrogate_parameterization" and s.strategy_id.startswith("component_surrogate_hsi_codesign"):
            component = "binary2phase" if "binary2phase" in s.strategy_id else "fresnel"
            return [ExperimentDesignCandidate(
                design_id=f"component_surrogate_{component}_hsi_codesign_design",
                objective=f"Run {component} component surrogate PSF through synthetic HSI reconstruction loss",
                backend_id="component_surrogate_psf",
                task_type="component_surrogate_hsi_codesign",
                spec_payload={
                    "component": component,
                    "dataset": "synthetic",
                    "steps": 3,
                    "bands": 4,
                    "image_size": 16,
                    "psf_size": 9,
                },
                expected_evidence_level="component_surrogate_hsi_codesign",
                expected_failure_modes=["surrogate_graph_disconnected"],
                source_strategy_id=s.strategy_id,
                strategy_family=s.strategy_type,
                design_origin="evidence_strategy_reasoner",
                required_skills=s.required_skills,
                claim_ceiling="component_surrogate_hsi_codesign",
                estimated_runtime_sec=300,
                risk_level=s.risk,
                expected_handler_id="component_surrogate_hsi_codesign",
                handler_capability_id="component_surrogate_hsi_codesign",
                claim_boundary_notes=[
                    "Component surrogate HSI co-design only — not full GeoLens lens-level optimization",
                    "Synthetic HSI only — no real HSI performance claim",
                ],
            )]
        return []

    def export(self, designs: list[ExperimentDesignCandidate],
               output_path: str | Path | None = None) -> Path:
        path = Path(output_path or "workspace/reports/experiment_design_candidates.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [d.__dict__ for d in designs]
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return path

    def export_markdown(self, designs: list[ExperimentDesignCandidate],
                        output_path: str | Path | None = None) -> Path:
        path = Path(output_path or "workspace/reports/experiment_design_candidates.md")
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Experiment Design Candidates",
            "",
            f"**Total Designs:** {len(designs)}",
            "",
            "| # | Design | Backend | Task | Evidence | Risk | Runtime |",
            "|---|---|---|---|---|---|---|",
        ]
        for i, d in enumerate(designs, 1):
            lines.append(
                f"| {i} | {d.design_id} | {d.backend_id} | {d.task_type} | "
                f"{d.expected_evidence_level} | {d.risk_level} | {d.estimated_runtime_sec}s |"
            )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path


def _evidence_rank(level: str) -> int:
    """Return numeric rank for evidence level comparison. Higher = stronger evidence."""
    ranks = {
        "": 0,
        "requires_user_data": 0,
        "structured_unsupported": 0,
        "needs_followup": 0,
        "report_only": 1,
        "negative_result": 1,
        "mock_simulation": 2,
        "deeplens_integration_smoke": 3,
        "native_component_optimization": 4,
        "component_surrogate_hsi_codesign": 5,
        "native_hsi_proxy": 5,
        "native_full_reconstruction_proxy": 6,
        "lightweight_scientific_execution": 7,
        "sweep_analysis": 7,
        "native_lens_simulation": 8,
        "native_waveoptics_simulation": 9,
        "stable_native_lens_hsi_codesign": 10,
        "rollback_protected_native_lens_hsi": 11,
        "real_hsi_performance": 12,
        "real_hsi": 12,
    }
    return ranks.get(level, 0)
