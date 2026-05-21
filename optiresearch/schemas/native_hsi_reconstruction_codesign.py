"""Native HSI Reconstruction CoDesign schemas for Phase 21."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import Field, field_validator

from optiresearch.memory.schemas import StrictModel, make_deterministic_id

RECON_CODESIGN_SCHEMA_VERSION = "0.1-draft"

VALID_RECON_OPTICAL_COMPONENTS = ["Fresnel", "Binary2Phase", "GeoLensCooke"]
VALID_RECONSTRUCTORS = ["differentiable_linear", "tiny_cnn"]


def make_recon_codesign_id(optical_component: str, reconstructor: str) -> str:
    return make_deterministic_id("recon_codesign", optical_component, reconstructor)


class NativeHSIReconstructionCoDesignSpec(StrictModel):
    schema_version: str = Field(default=RECON_CODESIGN_SCHEMA_VERSION)
    run_id: str = Field(min_length=1)
    optical_component: str = Field(min_length=1)
    reconstructor: str = Field(min_length=1)
    dataset: str = Field(default="synthetic")
    bands: int = Field(default=31, ge=1, le=256)
    image_size: int = Field(default=32, ge=8, le=256)
    psf_size: int = Field(default=16, ge=4, le=128)
    batch_size: int = Field(default=2, ge=1, le=64)
    optical_lr: float = Field(default=1e-3, gt=0.0, le=1.0)
    recon_lr: float = Field(default=1e-3, gt=0.0, le=1.0)
    max_steps: int = Field(default=5, ge=1, le=100)
    optimize_optics: bool = Field(default=True)
    optimize_reconstructor: bool = Field(default=True)
    freeze_reconstructor_after_warmup: bool = Field(default=False)
    loss_weights: dict[str, float] = Field(
        default_factory=lambda: {"mse": 1.0, "spectral_angle": 0.05, "measurement_consistency": 0.1}
    )
    device: str = Field(default="cpu")
    strict_native: bool = Field(default=True)
    save_artifacts: bool = Field(default=True)
    allow_phase_to_fft_proxy: bool = Field(default=True)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("optical_component")
    @classmethod
    def component_must_be_valid(cls, value: str) -> str:
        if value not in VALID_RECON_OPTICAL_COMPONENTS:
            raise ValueError(
                f"optical_component must be one of {VALID_RECON_OPTICAL_COMPONENTS}, got {value!r}"
            )
        return value

    @field_validator("reconstructor")
    @classmethod
    def reconstructor_must_be_valid(cls, value: str) -> str:
        if value not in VALID_RECONSTRUCTORS:
            raise ValueError(
                f"reconstructor must be one of {VALID_RECONSTRUCTORS}, got {value!r}"
            )
        return value

    @field_validator("device")
    @classmethod
    def device_must_be_valid(cls, value: str) -> str:
        if value not in ("cpu", "cuda", "mps"):
            raise ValueError(f"device must be cpu, cuda, or mps, got {value!r}")
        return value


class NativeHSIReconstructionCoDesignResult(StrictModel):
    schema_version: str = Field(default=RECON_CODESIGN_SCHEMA_VERSION)
    run_id: str = Field(min_length=1)
    status: Literal["succeeded", "unsupported", "failed"]
    optical_component: str = Field(min_length=1)
    reconstructor: str = Field(default="")
    differentiable: bool = False
    full_reconstruction_loss_used: bool = False
    native_parameter_update: bool = False
    full_wave_optics: bool = False
    phase_to_fft_proxy_used: bool = True
    reconstruction_loss_before: Optional[float] = None
    reconstruction_loss_after: Optional[float] = None
    mse_before: Optional[float] = None
    mse_after: Optional[float] = None
    spectral_angle_before: Optional[float] = None
    spectral_angle_after: Optional[float] = None
    measurement_consistency_before: Optional[float] = None
    measurement_consistency_after: Optional[float] = None
    psnr_before: Optional[float] = None
    psnr_after: Optional[float] = None
    sam_before: Optional[float] = None
    sam_after: Optional[float] = None
    optical_gradient_norm: Optional[float] = None
    recon_gradient_norm: Optional[float] = None
    optical_parameters_changed: Optional[bool] = None
    recon_parameters_changed: Optional[bool] = None
    optimizer_step_executed: bool = False
    autograd_graph_exists: bool = False
    detached_tensors_detected: bool = False
    evidence_level: Optional[str] = None
    artifact_paths: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
