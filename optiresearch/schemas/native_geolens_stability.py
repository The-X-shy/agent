"""Native GeoLens Stability schema — extends stable native lens HSI co-design.

Adds multi-objective loss configuration, multi-metric rollback policy,
and stability-specific diagnostics to the base StableNativeLensHSI types.
"""

from __future__ import annotations

from typing import Optional

from pydantic import Field, field_validator

from optiresearch.schemas.stable_native_lens_hsi import (
    STABLE_NATIVE_LENS_SCHEMA_VERSION,
    StableNativeLensHSIResult,
    StableNativeLensHSISpec,
)


class NativeGeoLensStabilitySpec(StableNativeLensHSISpec):
    """Extended spec with stability-specific configuration."""

    schema_version: str = Field(default=STABLE_NATIVE_LENS_SCHEMA_VERSION)
    seed: int = Field(default=42, ge=0)
    optimizer_name: str = Field(default="adam")
    spectral_angle_weight: float = Field(default=0.2, ge=0.0, le=10.0)
    optical_grad_clip: float = Field(default=1.0, gt=0.0, le=10000.0)
    enable_rollback_policy: bool = Field(default=True)
    rollback_max_grad_norm: float = Field(default=5000.0, gt=0.0)
    rollback_sam_tolerance: float = Field(default=0.0, ge=0.0)
    rollback_allow_tradeoff: bool = Field(default=False)

    @field_validator("optimizer_name")
    @classmethod
    def optimizer_name_valid(cls, v):
        if v not in ("adam", "sgd"):
            raise ValueError(f"optimizer_name must be 'adam' or 'sgd', got {v!r}")
        return v


class NativeGeoLensStabilityResult(StableNativeLensHSIResult):
    """Extended result with stability diagnostics."""

    schema_version: str = Field(default=STABLE_NATIVE_LENS_SCHEMA_VERSION)
    spectral_angle_weight: float = 0.2
    seed: int = 42
    optimizer_name: str = "adam"
    rollback_policy_enabled: bool = True
    accepted_update_count: int = 0
    rollback_count: int = 0
    rollback_reasons: list[str] = Field(default_factory=list)
    grad_norm_mean: Optional[float] = None
    psf_energy_before: Optional[float] = None
    psf_energy_after: Optional[float] = None
    psf_centroid_shift: Optional[float] = None
    psf_width_shift: Optional[float] = None
    stability_score: Optional[float] = None
    loss_terms_final: dict[str, float] = Field(default_factory=dict)
    metric_tradeoff_summary: str = ""
    warnings: list[str] = Field(default_factory=list)
