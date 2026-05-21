"""Surface-level differentiable optimization probe schemas.

Defines structured specs and results for probing individual DeepLens surface
classes (Fresnel, Binary2Phase, etc.) for native autograd support.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import Field, field_validator

from optiresearch.memory.schemas import StrictModel, make_deterministic_id

SURFACE_OPT_SCHEMA_VERSION = "0.1-draft"

VALID_SURFACE_CLASSES = [
    "Fresnel", "Binary2", "Zernike", "Grating", "Pixel2D", "ThinLens",
    "Binary2Phase", "CubicPhase", "ZernikePhase", "PolyPhase", "GratingPhase",
    "FresnelPhase", "NURBSPhase", "QPhase", "VortexPhase",
]

VALID_SURFACE_OBJECTIVES = [
    "minimize_phase_variance",
    "match_target_phase",
    "parameter_sanity_check",
]

VALID_SURFACE_PROBE_STATUSES = ["succeeded", "unsupported", "failed"]


def make_surface_probe_id(surface_class: str, objective: str) -> str:
    return make_deterministic_id("surf_opt_probe", surface_class, objective)


class SurfaceOptimizationProbeSpec(StrictModel):
    schema_version: str = Field(default=SURFACE_OPT_SCHEMA_VERSION)
    probe_id: str = Field(min_length=1)
    surface_class: str = Field(min_length=1)
    objective: str = Field(default="parameter_sanity_check")
    max_steps: int = Field(default=5, ge=1, le=100)
    learning_rate: float = Field(default=1e-3, gt=0.0, le=1.0)
    device: str = Field(default="cpu")
    save_artifacts: bool = Field(default=True)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("surface_class")
    @classmethod
    def surface_must_be_valid(cls, value: str) -> str:
        if value not in VALID_SURFACE_CLASSES:
            raise ValueError(
                f"surface_class must be one of {VALID_SURFACE_CLASSES}, got {value!r}"
            )
        return value

    @field_validator("objective")
    @classmethod
    def objective_must_be_valid(cls, value: str) -> str:
        if value not in VALID_SURFACE_OBJECTIVES:
            raise ValueError(
                f"objective must be one of {VALID_SURFACE_OBJECTIVES}, got {value!r}"
            )
        return value


class SurfaceOptimizationProbeResult(StrictModel):
    schema_version: str = Field(default=SURFACE_OPT_SCHEMA_VERSION)
    probe_id: str = Field(min_length=1)
    status: Literal["succeeded", "unsupported", "failed"]
    surface_class: str = Field(min_length=1)
    objective: str = Field(default="")
    module_path: Optional[str] = None
    can_instantiate: bool = False
    has_get_optimizer_params: bool = False
    has_get_optimizer: bool = False
    trainable_params: list[str] = Field(default_factory=list)
    differentiable: bool = False
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
