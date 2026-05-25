"""DeepLens Design Strategy Registry for Phase 54."""

from optiresearch.schemas.deeplens_design_strategy import DeepLensDesignStrategy


class DeepLensDesignStrategyRegistry:
    def __init__(self):
        self._strategies: dict[str, DeepLensDesignStrategy] = {}
        self._register_builtins()

    def get(self, sid: str) -> DeepLensDesignStrategy | None:
        return self._strategies.get(sid)

    def list_all(self) -> list[DeepLensDesignStrategy]:
        return list(self._strategies.values())

    def list_enabled(self) -> list[DeepLensDesignStrategy]:
        return [s for s in self._strategies.values() if s.enabled]

    def find_by_family(self, family: str) -> list[DeepLensDesignStrategy]:
        return [s for s in self._strategies.values() if s.strategy_family == family]

    def find_for_diagnosis(self, failure_modes: list[str]) -> list[DeepLensDesignStrategy]:
        return [s for s in self._strategies.values()
                if s.enabled and any(m in s.compatible_diagnosis_failure_modes for m in failure_modes)]

    def _register_builtins(self):
        builtins = [
            DeepLensDesignStrategy(
                strategy_id="geolens_curriculum_probe",
                name="GeoLens Curriculum Probe",
                strategy_family="curriculum_learning",
                objective="Avoid direct full-parameter update by staged difficulty progression",
                compatible_diagnosis_failure_modes=["unstable_training", "no_parameter_change"],
                evidence_level="diagnostic_evidence",
                claim_ceiling="diagnostic_evidence",
                required_skills=["deeplens_curriculum_probe"],
                parameters={"curriculum_stages": 4, "max_steps": 3, "rollback_policy": "per_stage"},
                caveats=["Curriculum probe only — does not guarantee stable optical improvement"],
            ),
            DeepLensDesignStrategy(
                strategy_id="geolens_regularized_probe",
                name="GeoLens Regularized Probe",
                strategy_family="optical_regularization",
                objective="Add PSF energy/centroid/width regularization to dampen gradient spikes",
                compatible_diagnosis_failure_modes=["unstable_training"],
                evidence_level="diagnostic_evidence",
                claim_ceiling="diagnostic_evidence",
                required_skills=["deeplens_regularized_probe"],
                parameters={"regularization_terms": ["psf_energy", "psf_centroid", "psf_width"], "max_steps": 3},
                caveats=["Regularization probe — not a validated optical design improvement"],
            ),
            DeepLensDesignStrategy(
                strategy_id="geolens_surface_freeze_unfreeze_probe",
                name="GeoLens Surface Freeze/Unfreeze Probe",
                strategy_family="surface_freeze_unfreeze",
                objective="Identify trainable surface subsets and freeze unstable surfaces",
                compatible_diagnosis_failure_modes=["no_parameter_change", "gradient_flow_blocked"],
                evidence_level="diagnostic_evidence",
                claim_ceiling="diagnostic_evidence",
                required_skills=["deeplens_trainable_parameter_inspection"],
                caveats=["Surface inspection — does not confirm optical design improvement"],
            ),
            DeepLensDesignStrategy(
                strategy_id="component_first_fresnel_probe",
                name="Component-First Fresnel Probe",
                strategy_family="component_first",
                objective="Verify stable differentiable Fresnel component update before lens-level training",
                compatible_diagnosis_failure_modes=["gradient_flow_blocked", "unstable_training"],
                evidence_level="native_component_optimization",
                claim_ceiling="native_component_optimization",
                required_skills=["deeplens_component_first_probe"],
                parameters={"component": "fresnel"},
            ),
            DeepLensDesignStrategy(
                strategy_id="component_first_binary2phase_probe",
                name="Component-First Binary2Phase Probe",
                strategy_family="component_first",
                objective="Verify differentiable phase polynomial component parameter update",
                evidence_level="native_component_optimization",
                claim_ceiling="native_component_optimization",
                required_skills=["deeplens_component_first_probe"],
                parameters={"component": "binary2phase"},
            ),
            DeepLensDesignStrategy(
                strategy_id="ray_to_wave_progression_probe",
                name="Ray-to-Wave Progression Probe",
                strategy_family="ray_to_wave_progression",
                objective="Probe geometric path first; coherent wave only if autograd passes",
                evidence_level="diagnostic_evidence",
                claim_ceiling="diagnostic_evidence",
                required_skills=["deeplens_autograd_audit"],
                caveats=["Wave-optics claim requires autograd verification — not yet supported"],
            ),
            DeepLensDesignStrategy(
                strategy_id="diffractive_candidate_probe",
                name="Diffractive Candidate Probe",
                strategy_family="diffractive_probe",
                objective="Check DeepLens diffractive candidate availability and differentiability",
                evidence_level="diagnostic_evidence",
                claim_ceiling="diagnostic_evidence",
                required_skills=["deeplens_component_first_probe"],
                parameters={"component": "diffractive_candidate"},
            ),
            DeepLensDesignStrategy(
                strategy_id="report_geolens_negative_result",
                name="Report GeoLens Negative Result",
                strategy_family="negative_result_report",
                objective="Document GeoLensCooke full parameterization gradient instability",
                evidence_level="report_only",
                claim_ceiling="report_only",
                required_skills=["report_generation"],
            ),
        ]
        for s in builtins:
            self._strategies[s.strategy_id] = s


_registry: DeepLensDesignStrategyRegistry | None = None


def get_deeplens_design_strategy_registry() -> DeepLensDesignStrategyRegistry:
    global _registry
    if _registry is None:
        _registry = DeepLensDesignStrategyRegistry()
    return _registry
