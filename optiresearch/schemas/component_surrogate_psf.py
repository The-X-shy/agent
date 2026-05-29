"""Schemas for component surrogate PSF and HSI co-design."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from optiresearch.memory.schemas import StrictModel, make_deterministic_id

COMPONENT_SURROGATE_PSF_SCHEMA_VERSION = "0.1-draft"

ComponentType = Literal["fresnel", "binary2phase", "diffractive_candidate"]
ParameterInit = Literal["default", "probe_result", "manual"]
SurrogateModel = Literal[
    "gaussian_width",
    "phase_to_intensity_fft",
    "polynomial_phase",
    "hybrid_simple",
]


def make_component_surrogate_hsi_run_id(
    component_type: str,
    dataset: str = "synthetic",
    steps: int = 3,
) -> str:
    return make_deterministic_id("comp_sur_hsi", component_type, dataset, steps)


class ComponentSurrogatePSFSpec(StrictModel):
    schema_version: str = Field(default=COMPONENT_SURROGATE_PSF_SCHEMA_VERSION)
    component_type: ComponentType
    psf_size: int = Field(ge=3, le=129)
    band_count: int = Field(ge=1, le=256)
    wavelengths_nm: list[float] | None = None
    normalize_psf: bool = True
    device: str = "cpu"
    parameter_init: ParameterInit = "default"
    surrogate_model: SurrogateModel = "hybrid_simple"
    strict_native_component: bool = True
    allow_proxy_psf: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("device")
    @classmethod
    def device_must_be_supported(cls, value: str) -> str:
        if value not in ("cpu", "cuda", "mps"):
            raise ValueError(f"device must be cpu, cuda, or mps, got {value!r}")
        return value

    @model_validator(mode="after")
    def wavelengths_match_band_count(self) -> "ComponentSurrogatePSFSpec":
        if self.wavelengths_nm is not None and len(self.wavelengths_nm) != self.band_count:
            raise ValueError("wavelengths_nm length must match band_count")
        return self


class ComponentSurrogatePSFResult(StrictModel):
    schema_version: str = Field(default=COMPONENT_SURROGATE_PSF_SCHEMA_VERSION)
    component_type: str
    status: Literal["succeeded", "needs_followup", "failed"] = "failed"
    evidence_level: str = "diagnostic_evidence"
    claim_ceiling: str = "diagnostic_evidence"
    psf_shape: list[int] = Field(default_factory=list)
    psf_requires_grad: bool = False
    parameter_count: int = 0
    trainable_param_count: int = 0
    params_with_grad: int = 0
    grad_norm_max: float = 0.0
    component_parameter_changed: bool = False
    psf_energy: list[float] = Field(default_factory=list)
    psf_centroid: list[float] = Field(default_factory=list)
    psf_width: float | None = None
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    error_code: str | None = None
    error_message: str | None = None
    parameter_names: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    psf: Any | None = Field(default=None, exclude=True)
    component_parameters: Any | None = Field(default=None, exclude=True)


class ComponentSurrogateHSICoDesignSpec(StrictModel):
    schema_version: str = Field(default=COMPONENT_SURROGATE_PSF_SCHEMA_VERSION)
    run_id: str | None = None
    component_type: ComponentType
    dataset: str = "synthetic"
    steps: int = Field(default=3, ge=1, le=100)
    device: str = "cpu"
    band_count: int = Field(default=4, ge=1, le=64)
    image_size: int = Field(default=16, ge=8, le=128)
    psf_size: int = Field(default=9, ge=3, le=65)
    batch_size: int = Field(default=1, ge=1, le=16)
    component_lr: float = Field(default=0.25, gt=0.0, le=10.0)
    recon_lr: float = Field(default=0.05, gt=0.0, le=10.0)
    optimize_reconstructor: bool = True
    seed: int = 63
    save_artifacts: bool = True
    psf_spec: ComponentSurrogatePSFSpec | None = None
    loss_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "mse": 1.0,
            "spectral_angle": 0.01,
            "measurement_consistency": 0.0,
        }
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("device")
    @classmethod
    def device_must_be_supported(cls, value: str) -> str:
        if value not in ("cpu", "cuda", "mps"):
            raise ValueError(f"device must be cpu, cuda, or mps, got {value!r}")
        return value

    @field_validator("dataset")
    @classmethod
    def dataset_must_be_known(cls, value: str) -> str:
        if value not in ("synthetic", "local_fixture", "real_hsi"):
            raise ValueError("dataset must be synthetic, local_fixture, or real_hsi")
        return value

    @model_validator(mode="after")
    def populate_ids_and_psf_spec(self) -> "ComponentSurrogateHSICoDesignSpec":
        if self.run_id is None:
            self.run_id = make_component_surrogate_hsi_run_id(
                self.component_type, self.dataset, self.steps
            )
        if self.psf_spec is None:
            self.psf_spec = ComponentSurrogatePSFSpec(
                component_type=self.component_type,
                psf_size=self.psf_size,
                band_count=self.band_count,
                device=self.device,
            )
        return self


class ComponentSurrogateHSICoDesignResult(StrictModel):
    schema_version: str = Field(default=COMPONENT_SURROGATE_PSF_SCHEMA_VERSION)
    run_id: str
    component_type: str
    status: Literal["succeeded", "needs_followup", "failed"] = "failed"
    reconstruction_loss_before: float | None = None
    reconstruction_loss_after: float | None = None
    mse_before: float | None = None
    mse_after: float | None = None
    psnr_before: float | None = None
    psnr_after: float | None = None
    sam_before: float | None = None
    sam_after: float | None = None
    component_grad_norm_max: float = 0.0
    component_parameter_changed: bool = False
    psf_requires_grad: bool = False
    loss_requires_grad: bool = False
    evidence_level: str = "diagnostic_evidence"
    claim_ceiling: str = "diagnostic_evidence"
    artifacts: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    error_code: str | None = None
    error_message: str | None = None
    psf_summary: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
