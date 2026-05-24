"""Skill Registry v2 for Phase 36."""

from __future__ import annotations

from typing import Any, Optional

from optiresearch.skills.contracts import SkillSpec


class SkillRegistryV2:
    def __init__(self):
        self._skills: dict[str, SkillSpec] = {}
        self._register_builtins()

    def register_skill(self, spec: SkillSpec) -> None:
        self._skills[spec.skill_id] = spec

    def list_skills(self) -> list[SkillSpec]:
        return list(self._skills.values())

    def get(self, skill_id: str) -> Optional[SkillSpec]:
        return self._skills.get(skill_id)

    def find_by_backend(self, backend_id: str) -> list[SkillSpec]:
        return [s for s in self._skills.values() if backend_id in s.required_backends or not s.required_backends]

    def find_by_task(self, task_type: str) -> list[SkillSpec]:
        return [s for s in self._skills.values() if task_type in s.tags or task_type in s.skill_id]

    def inspect_skill(self, skill_id: str) -> dict[str, Any] | None:
        spec = self._skills.get(skill_id)
        if spec is None:
            return None
        return spec.model_dump(mode="json")

    def _register_builtins(self) -> None:
        builtins: list[dict[str, Any]] = [
            {"skill_id": "deeplens_native_geolens_hsi_codesign",
             "name": "DeepLens Native GeoLens HSI Co-Design",
             "description": "Run native GeoLens geometric HSI co-design with stable optimization",
             "required_backends": ["deeplens_geolens_geometric"],
             "produced_artifacts": ["result.json", "spec.json"],
             "evidence_level": "native_lens_simulation",
             "risk_level": "high", "timeout_sec": 3600,
             "allowed_execution_targets": ["local", "remote"],
             "tags": ["stable_lens_hsi_codesign", "geolens", "hsi"],
             "claim_implications": ["native_lens_simulation"]},
            {"skill_id": "native_geolens_stabilization_sweep",
             "name": "Native GeoLens Stabilization Sweep",
             "description": "Systematic sweep of optical_lr, grad_clip, trust_region for stabilization",
             "required_backends": ["deeplens_geolens_geometric"],
             "produced_artifacts": ["sweep_results.json", "best_config.json", "sweep_table.md"],
             "evidence_level": "sweep_analysis", "risk_level": "high", "timeout_sec": 7200,
             "tags": ["stabilization_sweep", "geolens"],
             "claim_implications": ["stable_optical_update", "negative_result"]},
            {"skill_id": "backend_probe",
             "name": "Backend Probe", "description": "Probe backend availability and capabilities",
             "required_backends": [], "produced_artifacts": ["probe_result.json"],
             "evidence_level": "smoke", "risk_level": "low", "timeout_sec": 600,
             "tags": ["backend_probe", "diagnostics"],
             "claim_implications": ["backend_availability"]},
            {"skill_id": "claim_check",
             "name": "Claim Gate Check",
             "description": "Check a proposed claim against evidence boundaries",
             "required_backends": [], "produced_artifacts": ["claim_decision.json"],
             "evidence_level": None, "risk_level": "low", "timeout_sec": 60,
             "tags": ["claim_check", "gate"],
             "claim_implications": ["claim_boundary"]},
            {"skill_id": "autograd_audit",
             "name": "Autograd Audit", "description": "Audit autograd graph for gradient breaks",
             "required_backends": ["deeplens_geolens_geometric"],
             "produced_artifacts": ["autograd_audit.json"],
             "evidence_level": "diagnostic", "risk_level": "low", "timeout_sec": 300,
             "tags": ["autograd", "diagnostics"],
             "claim_implications": ["differentiable_path"]},
            {"skill_id": "strategy_recommendation",
             "name": "Strategy Recommendation",
             "description": "Recommend next research strategy from evidence",
             "required_backends": [], "produced_artifacts": ["strategy_recommendation.json"],
             "evidence_level": None, "risk_level": "low", "timeout_sec": 120,
             "tags": ["strategy", "planning"],
             "claim_implications": []},
            {"skill_id": "report_generation",
             "name": "Report Generation", "description": "Generate research reports from experiment results",
             "required_backends": [], "produced_artifacts": ["report.md"],
             "evidence_level": None, "risk_level": "low", "timeout_sec": 60,
             "tags": ["report", "documentation"],
             "claim_implications": []},
            {"skill_id": "remote_execution",
             "name": "Remote Execution", "description": "Execute experiments on remote workers",
             "required_backends": [], "produced_artifacts": ["remote_job_result.json"],
             "evidence_level": None, "risk_level": "medium", "timeout_sec": 7200,
             "allowed_execution_targets": ["remote"],
             "tags": ["remote", "execution"],
             "claim_implications": ["cross_platform"]},
            {"skill_id": "lightweight_scientific_hsi_mse_only",
             "name": "Lightweight Scientific HSI (MSE-only)",
             "description": "Run lightweight MSE-only HSI co-design with synthetic HSI data and FFT-based PSF generation. No DeepLens required.",
             "required_backends": [],
             "produced_artifacts": ["result.json"],
             "evidence_level": "lightweight_scientific_execution",
             "risk_level": "low", "timeout_sec": 120,
             "allowed_execution_targets": ["local"],
             "tags": ["lightweight", "scientific", "mse_only", "hsi"],
             "claim_implications": ["lightweight_scientific_execution"]},
            {"skill_id": "param_reduction_sweep",
             "name": "Param Reduction Sweep (Lightweight)",
             "description": "Run lightweight param-reduction sweep with low-dimensional pseudo-optical parameter vectors on synthetic HSI. No DeepLens required.",
             "required_backends": [],
             "produced_artifacts": ["result.json"],
             "evidence_level": "lightweight_scientific_execution",
             "risk_level": "low", "timeout_sec": 120,
             "allowed_execution_targets": ["local"],
             "tags": ["lightweight", "scientific", "param_reduction", "sweep"],
             "claim_implications": ["lightweight_scientific_execution"]},
            {"skill_id": "evidence_registry_export",
             "name": "Evidence Registry Export",
             "description": "Export evidence registry for audit and reporting",
             "required_backends": [], "produced_artifacts": ["evidence_export.json"],
             "evidence_level": None, "risk_level": "low", "timeout_sec": 60,
             "tags": ["evidence", "audit"],
             "claim_implications": []},
        ]
        for b in builtins:
            self.register_skill(SkillSpec(**b))
