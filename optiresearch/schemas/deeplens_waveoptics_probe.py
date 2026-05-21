"""DeepLens Wave-Optics Probe schemas for Phase 22."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import Field, field_validator

from optiresearch.memory.schemas import StrictModel, make_deterministic_id

WAVEOPTICS_PROBE_SCHEMA_VERSION = "0.1-draft"

VALID_WAVEOPTICS_CANDIDATES = [
    "GeoLensCooke",
    "DiffractiveLens",
    "HybridLens",
    "FresnelWave",
    "Binary2PhaseWave",
    "CustomLensFile",
]
VALID_WAVEOPTICS_OBJECTIVES = [
    "minimize_psf_width",
    "match_target_psf",
    "minimize_hsi_reconstruction_loss",
]


def make_waveoptics_probe_id(candidate: str, objective: str) -> str:
    return make_deterministic_id("waveoptics_probe", candidate, objective)


class DeepLensWaveOpticsProbeSpec(StrictModel):
    schema_version: str = Field(default=WAVEOPTICS_PROBE_SCHEMA_VERSION)
    run_id: str = Field(min_length=1)
    candidate: str = Field(min_length=1)
    lens_file: Optional[str] = None
    objective: str = Field(min_length=1)
    psf_size: int = Field(default=32, ge=4, le=256)
    bands: int = Field(default=31, ge=1, le=256)
    image_size: int = Field(default=32, ge=8, le=256)
    max_steps: int = Field(default=3, ge=1, le=100)
    learning_rate: float = Field(default=1e-3, gt=0.0, le=1.0)
    device: str = Field(default="cpu")
    strict_waveoptics: bool = Field(default=True)
    allow_phase_to_fft_proxy: bool = Field(default=False)
    save_artifacts: bool = Field(default=True)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("candidate")
    @classmethod
    def candidate_valid(cls, v):
        if v not in VALID_WAVEOPTICS_CANDIDATES:
            raise ValueError(f"candidate must be one of {VALID_WAVEOPTICS_CANDIDATES}, got {v!r}")
        return v

    @field_validator("objective")
    @classmethod
    def objective_valid(cls, v):
        if v not in VALID_WAVEOPTICS_OBJECTIVES:
            raise ValueError(f"objective must be one of {VALID_WAVEOPTICS_OBJECTIVES}, got {v!r}")
        return v

    @field_validator("device")
    @classmethod
    def device_valid(cls, v):
        if v not in ("cpu", "cuda", "mps"):
            raise ValueError(f"device must be cpu, cuda, or mps, got {v!r}")
        return v


class DeepLensWaveOpticsProbeResult(StrictModel):
    schema_version: str = Field(default=WAVEOPTICS_PROBE_SCHEMA_VERSION)
    run_id: str = Field(min_length=1)
    status: Literal["succeeded", "unsupported", "failed"]
    candidate: str = Field(min_length=1)
    lens_file: Optional[str] = None
    full_wave_optics: bool = False
    phase_to_fft_proxy_used: bool = True
    differentiable: bool = False
    native_parameter_update: bool = False
    hsi_reconstruction_loss_used: bool = False
    loss_before: Optional[float] = None
    loss_after: Optional[float] = None
    optical_gradient_norm: Optional[float] = None
    optical_parameter_before: Optional[dict[str, Any]] = None
    optical_parameter_after: Optional[dict[str, Any]] = None
    optical_parameters_changed: Optional[bool] = None
    psf_requires_grad: bool = False
    autograd_graph_exists: bool = False
    detached_tensors_detected: bool = False
    numpy_break_detected: bool = False
    optimizer_step_executed: bool = False
    deeplens_native_wave_path: Optional[str] = None
    deeplens_native_psf_path: Optional[str] = None
    evidence_level: Optional[str] = None
    artifact_paths: list[str] = Field(default_factory=list)
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    caveats: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
