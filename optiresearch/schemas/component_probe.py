"""Component-level probe schemas for Phase 62.

Component probes validate individual DeepLens surface components (Fresnel,
Binary2Phase, diffractive candidates) at a lower level than lens-level
optimization.  These schemas carry component-semantic fields rather than
surface-class fields — the runtime layer maps between them.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from optiresearch.memory.schemas import StrictModel, make_deterministic_id

COMPONENT_PROBE_SCHEMA_VERSION = "0.1-draft"

VALID_COMPONENTS = ("fresnel", "binary2phase", "diffractive")

VALID_COMPONENT_OBJECTIVES = (
    "parameter_sanity_check",
    "minimize_phase_variance",
    "match_target_phase",
)


class ComponentProbeSpec(StrictModel):
    schema_version: str = Field(default=COMPONENT_PROBE_SCHEMA_VERSION)
    probe_id: str = Field(min_length=1)
    component: str = Field(min_length=1)
    objective: str = Field(default="parameter_sanity_check")
    max_steps: int = Field(default=5, ge=1, le=100)
    learning_rate: float = Field(default=1e-3, gt=0)
    device: str = Field(default="cpu")
    save_artifacts: bool = Field(default=True)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ComponentProbeResult(StrictModel):
    schema_version: str = Field(default=COMPONENT_PROBE_SCHEMA_VERSION)
    probe_id: str = Field(min_length=1)
    component: str = Field(min_length=1)
    status: Literal["succeeded", "needs_followup", "structured_unavailable", "failed"] = "failed"
    surface_class: str | None = None
    backend_id: str = ""
    module_path: str | None = None
    can_instantiate: bool = False
    has_get_optimizer: bool = False
    has_get_optimizer_params: bool = False
    parameter_count: int = 0
    trainable_param_count: int = 0
    trainable_param_names: list[str] = Field(default_factory=list)
    params_with_grad: int = 0
    zero_gradient_parameters: list[str] = Field(default_factory=list)
    differentiable: bool = False
    autograd_graph_exists: bool = False
    parameters_changed: bool = False
    loss_before: float | None = None
    loss_after: float | None = None
    gradient_norm: float | None = None
    parameter_norm_before: float | None = None
    parameter_norm_after: float | None = None
    optimizer_class: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    evidence_level: str = "diagnostic_evidence"
    claim_ceiling: str = "diagnostic_evidence"
    checked_component_candidates: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


def make_component_probe_id(component: str, objective: str = "parameter_sanity_check") -> str:
    return make_deterministic_id("comp_probe", component, objective)
