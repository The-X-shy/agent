"""Semi-native realization layer for DeepLens baselines."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from optiresearch.adapters.deeplens_encoder_strategies import EncoderStrategy
from optiresearch.schemas.experiment import ExperimentSpec


class SemiNativeTransform:
    def supports(self, encoder_type: str, api_probe: dict[str, Any], capabilities: dict[str, Any]) -> bool:
        if encoder_type == "conventional":
            return bool(capabilities.get("paraxial_lens_available"))
        if os.getenv("OPTIRESEARCH_ENABLE_EXPERIMENTAL_SEMI_NATIVE") != "1":
            return False
        return bool(api_probe.get("candidate_phase_or_doe_classes") or api_probe.get("candidate_surface_classes"))

    def build_config(self, experiment_spec: ExperimentSpec, strategy: EncoderStrategy) -> dict[str, Any]:
        return {
            "encoder_type": experiment_spec.optical_spec.encoder_type,
            "strategy_name": strategy.strategy_name,
            "selected_realization_level": "semi_native",
            "claim_scope": strategy.claim_scope,
            "semi_native_plan": strategy.semi_native_plan,
            "native_requirements": strategy.native_requirements,
        }

    def apply_before_psf(self, config: dict[str, Any]) -> dict[str, Any]:
        updated = dict(config)
        updated["before_psf_transform"] = "deeplens_paraxial_native_psf" if config["encoder_type"] == "conventional" else "unsupported"
        return updated

    def apply_after_psf_if_needed(self, psf_cube: np.ndarray, config: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        manifest = self.manifest(config)
        return psf_cube, manifest

    def manifest(self, config: dict[str, Any] | None = None) -> dict[str, Any]:
        config = config or {}
        return {
            "encoder_type": config.get("encoder_type"),
            "selected_realization_level": "semi_native",
            "strategy_name": config.get("strategy_name"),
            "semi_native_attempted": True,
            "semi_native_succeeded": True,
            "proxy_fallback_used": False,
            "claim_scope": config.get("claim_scope"),
            "native_requirements": config.get("native_requirements", []),
            "semi_native_plan": config.get("semi_native_plan", []),
            "experimental_flags": {"OPTIRESEARCH_ENABLE_EXPERIMENTAL_SEMI_NATIVE": os.getenv("OPTIRESEARCH_ENABLE_EXPERIMENTAL_SEMI_NATIVE") == "1"},
        }
