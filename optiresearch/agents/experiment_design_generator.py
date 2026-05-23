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
    required_skills: list[str] = field(default_factory=list)
    claim_ceiling: str = ""
    estimated_runtime_sec: int = 600
    risk_level: str = "low"


class ExperimentDesignGenerator:
    def generate_designs(
        self,
        strategies: list[CandidateStrategy],
        constraints: dict[str, Any] | None = None,
    ) -> list[ExperimentDesignCandidate]:
        designs: list[ExperimentDesignCandidate] = []
        for s in strategies:
            designs.extend(self._generate_for_strategy(s))
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
                backend_id="deeplens_geolens_coherent",
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
