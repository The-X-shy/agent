"""Stable Native Lens-Simulation HSI Co-Design schemas for Phase 23."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import Field, field_validator

from optiresearch.memory.schemas import StrictModel, make_deterministic_id

STABLE_NATIVE_LENS_SCHEMA_VERSION = "0.1-draft"
VALID_STABLE_CANDIDATES = ["GeoLensCooke", "DiffractiveLens", "HybridLens"]
VALID_STABLE_RECONSTRUCTORS = ["differentiable_linear", "tiny_cnn"]


def make_stable_lens_id(candidate: str, reconstructor: str) -> str:
    return make_deterministic_id("stable_lens_hsi", candidate, reconstructor)


class StableNativeLensHSISpec(StrictModel):
    schema_version: str = Field(default=STABLE_NATIVE_LENS_SCHEMA_VERSION)
    run_id: str = Field(min_length=1)
    candidate: str = Field(min_length=1)
    reconstructor: str = Field(min_length=1)
    dataset: str = Field(default="synthetic")
    max_steps: int = Field(default=10, ge=1, le=100)
    optical_lr: float = Field(default=1e-6, gt=0.0, le=1e-2)
    recon_lr: float = Field(default=1e-3, gt=0.0, le=1e-1)
    optical_grad_clip: float = Field(default=1.0, gt=0.0, le=100.0)
    recon_grad_clip: float = Field(default=5.0, gt=0.0, le=100.0)
    optical_warmup_steps: int = Field(default=3, ge=0, le=50)
    optical_update_interval: int = Field(default=1, ge=1, le=10)
    rollback_on_loss_increase: bool = Field(default=True)
    accept_if_loss_delta_below: float = Field(default=0.0)
    psf_energy_reg_weight: float = Field(default=0.1, ge=0.0, le=10.0)
    psf_centroid_reg_weight: float = Field(default=0.1, ge=0.0, le=10.0)
    psf_width_reg_weight: float = Field(default=0.05, ge=0.0, le=10.0)
    optical_param_l2_weight: float = Field(default=1e-4, ge=0.0, le=1.0)
    max_optical_param_delta: float = Field(default=1e-3, gt=0.0, le=1.0)
    trust_region_enabled: bool = Field(default=False)
    rollback_on_psf_instability: bool = Field(default=False)
    max_psf_energy_delta: float = Field(default=0.1, gt=0.0, le=10.0)
    max_psf_centroid_delta: float = Field(default=1.0, gt=0.0, le=50.0)
    max_psf_width_delta: float = Field(default=2.0, gt=0.0, le=50.0)
    accept_tolerance: float = Field(default=0.0)
    loss_weights: dict[str, float] = Field(default_factory=lambda: {"mse": 1.0, "spectral_angle": 0.05, "measurement_consistency": 0.1})
    bands: int = Field(default=4, ge=1, le=256)
    image_size: int = Field(default=16, ge=8, le=256)
    psf_size: int = Field(default=15, ge=4, le=256)
    device: str = Field(default="cpu")
    full_wave_optics: bool = Field(default=False)
    phase_to_fft_proxy_used: bool = Field(default=False)
    save_artifacts: bool = Field(default=True)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("candidate")
    @classmethod
    def candidate_valid(cls, v):
        if v not in VALID_STABLE_CANDIDATES:
            raise ValueError(f"candidate must be one of {VALID_STABLE_CANDIDATES}, got {v!r}")
        return v

    @field_validator("reconstructor")
    @classmethod
    def recon_valid(cls, v):
        if v not in VALID_STABLE_RECONSTRUCTORS:
            raise ValueError(f"reconstructor must be one of {VALID_STABLE_RECONSTRUCTORS}, got {v!r}")
        return v

    @field_validator("device")
    @classmethod
    def device_valid(cls, v):
        if v not in ("cpu", "cuda", "mps"):
            raise ValueError(f"device must be cpu, cuda, or mps, got {v!r}")
        return v


class StableNativeLensHSIResult(StrictModel):
    schema_version: str = Field(default=STABLE_NATIVE_LENS_SCHEMA_VERSION)
    run_id: str = Field(min_length=1)
    status: Literal["succeeded", "unsupported", "failed"]
    candidate: str = Field(min_length=1)
    reconstructor: str = Field(default="")
    reconstruction_loss_before: Optional[float] = None
    reconstruction_loss_after: Optional[float] = None
    best_reconstruction_loss: Optional[float] = None
    accepted_update_count: int = 0
    rejected_update_count: int = 0
    rollback_count: int = 0
    optical_gradient_norm_max: Optional[float] = None
    optical_gradient_norm_mean: Optional[float] = None
    recon_gradient_norm_max: Optional[float] = None
    recon_gradient_norm_mean: Optional[float] = None
    trainable_param_count: int = 0
    params_with_grad: int = 0
    graph_connected: bool = False
    psf_requires_grad: bool = False
    loss_requires_grad: bool = False
    optical_parameters_changed: Optional[bool] = None
    component_parameter_changed: Optional[bool] = None
    optical_parameter_delta_max: Optional[float] = None
    psf_energy_delta: Optional[float] = None
    psf_centroid_delta: Optional[float] = None
    psf_width_delta: Optional[float] = None
    mse_before: Optional[float] = None
    mse_after: Optional[float] = None
    psnr_before: Optional[float] = None
    psnr_after: Optional[float] = None
    sam_before: Optional[float] = None
    sam_after: Optional[float] = None
    stable_training_succeeded: bool = False
    full_wave_optics: bool = False
    phase_to_fft_proxy_used: bool = False
    deeplens_native_psf_path: Optional[str] = None
    evidence_level: Optional[str] = None
    optimizer_step_executed: bool = False
    artifact_paths: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    rollback_trace: list[dict[str, Any]] = Field(default_factory=list)
    trust_region_activated: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
