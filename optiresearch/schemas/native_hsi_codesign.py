"""Native Optical-HSI CoDesign schemas for Phase 20."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import Field, field_validator

from optiresearch.memory.schemas import StrictModel, make_deterministic_id

HSI_CODESIGN_SCHEMA_VERSION = "0.1-draft"

VALID_OPTICAL_COMPONENTS = ["Fresnel", "Binary2Phase", "GeoLensCooke"]

VALID_HSI_OBJECTIVES = [
    "minimize_hsi_proxy_loss",
    "maximize_reconstruction_score",
    "minimize_spectral_mse",
    "minimize_measurement_consistency_loss",
]

VALID_HSI_CODESIGN_STATUSES = ["succeeded", "unsupported", "failed"]


def make_hsi_codesign_id(optical_component: str, objective: str) -> str:
    return make_deterministic_id("native_hsi_codesign", optical_component, objective)


class NativeOpticalHSICoDesignSpec(StrictModel):
    schema_version: str = Field(default=HSI_CODESIGN_SCHEMA_VERSION)
    run_id: str = Field(min_length=1)
    optical_component: str = Field(min_length=1)
    objective: str = Field(min_length=1)
    dataset: str = Field(default="synthetic")
    bands: int = Field(default=31, ge=1, le=256)
    image_size: int = Field(default=32, ge=8, le=256)
    psf_size: int = Field(default=16, ge=4, le=128)
    max_steps: int = Field(default=3, ge=1, le=100)
    learning_rate: float = Field(default=1e-3, gt=0.0, le=1.0)
    device: str = Field(default="cpu")
    allow_proxy_loss: bool = Field(default=True)
    strict_native: bool = Field(default=True)
    save_artifacts: bool = Field(default=True)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("optical_component")
    @classmethod
    def component_must_be_valid(cls, value: str) -> str:
        if value not in VALID_OPTICAL_COMPONENTS:
            raise ValueError(
                f"optical_component must be one of {VALID_OPTICAL_COMPONENTS}, got {value!r}"
            )
        return value

    @field_validator("objective")
    @classmethod
    def objective_must_be_valid(cls, value: str) -> str:
        if value not in VALID_HSI_OBJECTIVES:
            raise ValueError(
                f"objective must be one of {VALID_HSI_OBJECTIVES}, got {value!r}"
            )
        return value

    @field_validator("device")
    @classmethod
    def device_must_be_valid(cls, value: str) -> str:
        if value not in ("cpu", "cuda", "mps"):
            raise ValueError(f"device must be cpu, cuda, or mps, got {value!r}")
        return value


class NativeOpticalHSICoDesignResult(StrictModel):
    schema_version: str = Field(default=HSI_CODESIGN_SCHEMA_VERSION)
    run_id: str = Field(min_length=1)
    status: Literal["succeeded", "unsupported", "failed"]
    optical_component: str = Field(min_length=1)
    objective: str = Field(default="")
    differentiable: bool = False
    native_parameter_update: bool = False
    hsi_loss_before: Optional[float] = None
    hsi_loss_after: Optional[float] = None
    reconstruction_metric_before: Optional[float] = None
    reconstruction_metric_after: Optional[float] = None
    gradient_norm: Optional[float] = None
    parameters_changed: Optional[bool] = None
    optimizer_step_executed: bool = False
    autograd_break_detected: bool = False
    evidence_level: Optional[str] = None
    artifact_paths: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
