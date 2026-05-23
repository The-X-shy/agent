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
            required_backend="deeplens_geolens_coherent",
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

    def get_top_strategy(self) -> Optional[CandidateStrategy]:
        if not self._strategies:
            return None
        priority_order = [
            "report_negative_result",
            "objective_redesign",
            "alternative_parameterization",
            "backend_switch",
            "waveoptics_probe",
            "parameter_reduction",
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
