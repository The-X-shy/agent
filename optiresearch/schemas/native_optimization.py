"""Native differentiable optimization probe schemas.

Defines structured specifications and results for probing whether
DeepLens lens classes support true autograd-based optical optimization:
parameter -> PSF simulation -> scalar loss -> backward -> optimizer.step
-> parameter change.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import Field, field_validator

from optiresearch.memory.schemas import StrictModel, make_deterministic_id

NATIVE_OPT_SCHEMA_VERSION = "0.1-draft"

VALID_LENS_CLASSES = [
    "ParaxialLens",
    "GeoLens",
    "DiffractiveLens",
    "HybridLens",
    "PSFNetLens",
]

VALID_OBJECTIVES = [
    "minimize_psf_width",
    "maximize_center_intensity",
    "match_target_psf",
    "hsi_reconstruction_loss",
]

VALID_REALIZATION_LEVELS = [
    "native",
    "semi_native",
    "adapter_proxy",
    "unavailable",
]

VALID_PROBE_STATUSES = [
    "succeeded",
    "unsupported",
    "failed",
]


def make_probe_id(lens_class: str, objective: str) -> str:
    return make_deterministic_id("native_opt_probe", lens_class, objective)


class NativeOptimizationProbeSpec(StrictModel):
    """Specification for a single native optimization probe run."""

    schema_version: str = Field(default=NATIVE_OPT_SCHEMA_VERSION)
    probe_id: str = Field(min_length=1)
    lens_class: str = Field(min_length=1)
    objective: str = Field(min_length=1)
    max_steps: int = Field(default=2, ge=1, le=100)
    learning_rate: float = Field(default=1e-3, gt=0.0, le=1.0)
    device: str = Field(default="cpu")
    strict_native: bool = Field(default=True)
    allow_adapter_proxy: bool = Field(default=False)
    save_artifacts: bool = Field(default=True)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("lens_class")
    @classmethod
    def lens_class_must_be_valid(cls, value: str) -> str:
        if value not in VALID_LENS_CLASSES:
            raise ValueError(
                f"lens_class must be one of {VALID_LENS_CLASSES}, got {value!r}"
            )
        return value

    @field_validator("objective")
    @classmethod
    def objective_must_be_valid(cls, value: str) -> str:
        if value not in VALID_OBJECTIVES:
            raise ValueError(
                f"objective must be one of {VALID_OBJECTIVES}, got {value!r}"
            )
        return value

    @field_validator("device")
    @classmethod
    def device_must_be_valid(cls, value: str) -> str:
        if value not in ("cpu", "cuda", "mps"):
            raise ValueError(f"device must be cpu, cuda, or mps, got {value!r}")
        return value


class NativeOptimizationProbeResult(StrictModel):
    """Structured result of a native optimization probe run."""

    schema_version: str = Field(default=NATIVE_OPT_SCHEMA_VERSION)
    probe_id: str = Field(min_length=1)
    status: Literal["succeeded", "unsupported", "failed"]
    lens_class: str = Field(min_length=1)
    objective: str = Field(default="")
    realization_level: Literal["native", "semi_native", "adapter_proxy", "unavailable"]
    differentiable: bool = False
    native_parameter_update: bool = False
    autograd_graph_exists: bool = False
    loss_before: Optional[float] = None
    loss_after: Optional[float] = None
    parameter_norm_before: Optional[float] = None
    parameter_norm_after: Optional[float] = None
    gradient_norm: Optional[float] = None
    parameters_changed: Optional[bool] = None
    optimizer_class: Optional[str] = None
    artifact_paths: list[str] = Field(default_factory=list)
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    caveats: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


def build_default_paraxial_psf_width_probe() -> NativeOptimizationProbeSpec:
    """Build a default probe spec for ParaxialLens with minimize_psf_width."""
    lens_class = "ParaxialLens"
    objective = "minimize_psf_width"
    return NativeOptimizationProbeSpec(
        probe_id=make_probe_id(lens_class, objective),
        lens_class=lens_class,
        objective=objective,
        max_steps=2,
        learning_rate=1e-3,
        device="cpu",
        strict_native=True,
        allow_adapter_proxy=False,
        save_artifacts=True,
    )
