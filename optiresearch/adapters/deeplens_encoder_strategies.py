"""Encoder-specific DeepLens strategy registry.

Phase 7 keeps the distinction between native DeepLens physics and
adapter-level proxy behavior explicit. The strategies below define how each
frozen ExperimentSpec encoder family is realized by the current backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


RealizationLevel = Literal["native", "semi_native", "adapter_proxy", "postprocess_proxy", "unsupported"]


@dataclass(frozen=True)
class EncoderStrategy:
    encoder_type: str
    strategy_name: str
    backend: str
    realization_level: RealizationLevel
    description: str
    expected_effects: dict[str, float]
    native_requirements: list[str] = field(default_factory=list)
    semi_native_plan: list[str] = field(default_factory=list)
    proxy_fallback: str = "adapter_proxy"
    validation_requirements: list[str] = field(default_factory=list)
    claim_scope: str = ""
    unsupported_fields: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


PHYSICAL_VALIDATION_LEVEL = "deeplens_base_psf_plus_adapter_proxy"


_STRATEGIES: dict[str, EncoderStrategy] = {
    "conventional": EncoderStrategy(
        encoder_type="conventional",
        strategy_name="conventional_paraxial_depth_proxy",
        backend="deeplens",
        realization_level="semi_native",
        description="Uses DeepLens ParaxialLens base PSF and a weak depth-variation proxy as the conventional baseline.",
        expected_effects={
            "psf_depth_similarity": 0.46,
            "spectral_separability": 0.06,
            "deeplens_mtf_mean": 0.72,
            "deeplens_energy_efficiency": 0.9,
        },
        native_requirements=["ParaxialLens PSF generation"],
        semi_native_plan=["Use DeepLens ParaxialLens as the baseline geometric lens behavior."],
        proxy_fallback="conventional_paraxial_depth_proxy",
        validation_requirements=["Verify ParaxialLens PSF artifact shape and metrics."],
        claim_scope="baseline DeepLens ParaxialLens behavior",
        unsupported_fields=[
            {
                "field": "native_conventional_encoder_design",
                "reason": "No dedicated native conventional encoder design binding is exposed in Phase 7.",
            }
        ],
    ),
    "achromatic": EncoderStrategy(
        encoder_type="achromatic",
        strategy_name="achromatic_shared_wavelength_proxy",
        backend="deeplens",
        realization_level="adapter_proxy",
        description="Reduces wavelength-dependent PSF variation through adapter-level wavelength normalization.",
        expected_effects={
            "psf_depth_similarity": 0.74,
            "spectral_separability": 0.05,
            "deeplens_mtf_mean": 0.8,
            "deeplens_energy_efficiency": 0.87,
        },
        native_requirements=["multi-wavelength optical model", "achromatic element or constraint"],
        semi_native_plan=["Analyze wavelength-dependent PSF variation when multi-wavelength generation is available."],
        proxy_fallback="achromatic_shared_wavelength_proxy",
        validation_requirements=["Do not claim full achromatic design without native element support."],
        claim_scope="multi-wavelength stability analysis, not full achromatic lens design",
        unsupported_fields=[
            {
                "field": "native_achromatic_lens_api",
                "reason": "Achromatic behavior is represented by an adapter proxy, not a native DeepLens achromat.",
            }
        ],
    ),
    "edof": EncoderStrategy(
        encoder_type="edof",
        strategy_name="edof_depth_smoothing_proxy",
        backend="deeplens",
        realization_level="adapter_proxy",
        description="Improves depth invariance by smoothing the DeepLens base PSF response across depths.",
        expected_effects={
            "psf_depth_similarity": 0.87,
            "spectral_separability": 0.14,
            "deeplens_mtf_mean": 0.58,
            "deeplens_energy_efficiency": 0.79,
        },
        native_requirements=["phase mask, DOE, or surface perturbation before PSF generation"],
        semi_native_plan=["Use lens-side perturbation if DeepLens exposes phase/surface classes."],
        proxy_fallback="edof_depth_smoothing_proxy",
        validation_requirements=["Confirm PSF shaping occurs before PSF generation."],
        claim_scope="EDOF-like PSF shaping, not fabricated EDOF optics",
        unsupported_fields=[
            {
                "field": "native_edof_phase_mask",
                "reason": "EDOF phase optimization is not yet mapped to a native DeepLens design.",
            }
        ],
    ),
    "chromatic_coded": EncoderStrategy(
        encoder_type="chromatic_coded",
        strategy_name="chromatic_spatial_modulation_proxy",
        backend="deeplens",
        realization_level="adapter_proxy",
        description="Applies wavelength-dependent spatial modulation to increase spectral separability.",
        expected_effects={
            "psf_depth_similarity": 0.53,
            "spectral_separability": 0.68,
            "deeplens_mtf_mean": 0.5,
            "deeplens_energy_efficiency": 0.72,
        },
        native_requirements=["wavelength-dependent phase or surface behavior before metric computation"],
        semi_native_plan=["Use wavelength-aware phase/surface class if available."],
        proxy_fallback="chromatic_spatial_modulation_proxy",
        validation_requirements=["Confirm coding is not output-only postprocessing."],
        claim_scope="spectral coding proxy",
        unsupported_fields=[
            {
                "field": "native_chromatic_coded_surface",
                "reason": "Chromatic coding is currently represented by an adapter-level modulation proxy.",
            }
        ],
    ),
    "controlled_chromatic_edof": EncoderStrategy(
        encoder_type="controlled_chromatic_edof",
        strategy_name="controlled_chromatic_edof_joint_proxy",
        backend="deeplens",
        realization_level="adapter_proxy",
        description="Combines depth-stabilizing smoothing with controlled wavelength coding for joint proxy behavior.",
        expected_effects={
            "psf_depth_similarity": 0.84,
            "spectral_separability": 0.59,
            "deeplens_mtf_mean": 0.64,
            "deeplens_energy_efficiency": 0.82,
        },
        native_requirements=["depth-stability mechanism", "wavelength coding mechanism", "joint optimization support"],
        semi_native_plan=["Combine lens-side depth-stability and wavelength coding only when both mechanisms are exposed."],
        proxy_fallback="controlled_chromatic_edof_joint_proxy",
        validation_requirements=["Require native optimization before final EDOF-HSI claims."],
        claim_scope="joint depth-spectral coding proxy",
        unsupported_fields=[
            {
                "field": "native_controlled_chromatic_edof_optimization",
                "reason": "Joint EDOF-HSI optimization is not yet bound to native DeepLens optimization.",
            }
        ],
    ),
}


def list_deeplens_encoder_strategies() -> list[EncoderStrategy]:
    return list(_STRATEGIES.values())


def get_deeplens_encoder_strategy(encoder_type: str) -> EncoderStrategy:
    try:
        return _STRATEGIES[encoder_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported DeepLens encoder strategy: {encoder_type}") from exc


def choose_best_realization_level(
    encoder_type: str,
    deeplens_capabilities: dict[str, Any],
    api_probe: dict[str, Any],
    requested: str = "auto",
) -> str:
    if requested == "adapter_proxy":
        return "adapter_proxy"
    if requested == "native":
        return "native" if deeplens_capabilities.get("encoder_specific_native_available") else "adapter_proxy"
    if requested in {"auto", "semi_native"}:
        if encoder_type == "conventional" and deeplens_capabilities.get("paraxial_lens_available"):
            return "semi_native"
        experimental = bool(api_probe.get("experimental_semi_native_enabled"))
        has_phase = bool(api_probe.get("candidate_phase_or_doe_classes") or api_probe.get("candidate_surface_classes"))
        if experimental and has_phase:
            return "semi_native"
    return "adapter_proxy"


def explain_realization_level(encoder_type: str, level: str) -> str:
    strategy = get_deeplens_encoder_strategy(encoder_type)
    if level == "semi_native":
        return f"{encoder_type} semi-native: {strategy.claim_scope}"
    if level == "adapter_proxy":
        return f"{encoder_type} adapter-proxy fallback: {strategy.proxy_fallback}"
    if level == "native":
        return f"{encoder_type} native realization requires: {', '.join(strategy.native_requirements)}"
    return f"{encoder_type} realization level: {level}"


def strategy_to_metadata(strategy: EncoderStrategy, selected_realization_level: str | None = None) -> dict[str, Any]:
    level = selected_realization_level or strategy.realization_level
    proxy_applied = level in {"adapter_proxy", "postprocess_proxy"}
    semi_native = level == "semi_native"
    return {
        "backend": strategy.backend,
        "backend_capability_level": "proxy" if proxy_applied else level,
        "encoder_type": strategy.encoder_type,
        "encoder_behavior_realized": level != "unsupported",
        "encoder_behavior_realization_level": level,
        "selected_realization_level": level,
        "physical_validation_level": "baseline DeepLens ParaxialLens behavior" if semi_native else (PHYSICAL_VALIDATION_LEVEL if proxy_applied else level),
        "proxy_transform_applied": proxy_applied,
        "proxy_transform_name": strategy.strategy_name if proxy_applied else None,
        "semi_native_attempted": semi_native,
        "semi_native_succeeded": semi_native,
        "proxy_fallback_used": False,
        "claim_scope": strategy.claim_scope,
        "native_requirements": strategy.native_requirements,
        "semi_native_plan": strategy.semi_native_plan,
        "validation_requirements": strategy.validation_requirements,
        "unsupported_fields": strategy.unsupported_fields,
        "wavelength_aware_psf_contract": True,
        "native_wavelength_physics": level == "native",
        "hsi_forward_compatible_psf_contract": level != "unsupported",
        **strategy.metadata,
    }
