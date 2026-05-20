"""Optimization specification for optical–HSI co-design.

Defines concrete optical variables that can be optimized:
  - phase_mask_strength: wavefront modulation amplitude
  - doe_grating_period: diffractive optical element grating spacing
  - surface_curvature: focus depth control
  - chromatic_shift: wavelength-dependent PSF shift
  - depth_variation: depth-dependent PSF variation

These variables parameterize the PSF generation and can be
iteratively updated by an agent-driven co-design loop.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import Field

from optiresearch.memory.schemas import StrictModel, make_deterministic_id


class OpticalVariable(StrictModel):
    """A single optimizable optical parameter."""

    name: str
    description: str = ""
    min_value: float = 0.0
    max_value: float = 1.0
    current_value: float = 0.5
    step_size: float = 0.05
    unit: str = ""


class CoDesignState(StrictModel):
    """State of a single co-design iteration."""

    iteration: int
    optical_variables: dict[str, float]
    psf_metrics: dict[str, Any] = {}
    hsi_metrics: dict[str, Any] = {}
    reconstruction_score: float = 0.0
    loss_value: float = 0.0
    improvement_from_previous: Optional[float] = None
    agent_decision: str = ""
    agent_rationale: str = ""


class OptimizationSpec(StrictModel):
    """Specification for optical-HSI co-design optimization.

    Extends the original draft with concrete optical variable definitions
    and co-design specific configuration.
    """

    schema_version: str = "0.2-draft"
    optimization_id: str = Field(min_length=1)
    objective: str = ""
    target_metrics: list[str] = Field(default_factory=list)
    optical_variables: list[OpticalVariable] = Field(default_factory=list)
    loss_terms: list[dict[str, Any]] = Field(default_factory=list)
    variables: list[dict[str, Any]] = Field(default_factory=list)
    constraints: dict[str, Any] = Field(default_factory=dict)
    max_iterations: int = Field(default=20, ge=1)
    budget: dict[str, Any] = Field(default_factory=dict)
    backend: str = "mock_deeplens"
    encoder_type: str = "controlled_chromatic_edof"
    reconstructor_type: str = "optical_conditioned_linear"
    forward_mode: str = "depth_spectral_coded"
    dataset: str = "synthetic"
    requires_native_support: bool = False
    llm_provider: str = "mock"
    psf_source: str = "parameterized_mock"
    fallback_policy: str = "fallback_to_mock"
    strict_deeplens: bool = False
    stopping_criteria: list[str] = Field(default_factory=lambda: ["max_iterations", "no_improvement", "convergence"])
    convergence_threshold: float = 0.001
    metadata: dict[str, Any] = Field(default_factory=dict)


def build_default_optical_variables() -> list[OpticalVariable]:
    """Build standard optical variables for co-design optimization."""
    return [
        OpticalVariable(
            name="phase_mask_strength",
            description="Wavefront modulation amplitude (controls PSF shape coding)",
            min_value=0.0,
            max_value=1.0,
            current_value=0.5,
            step_size=0.1,
        ),
        OpticalVariable(
            name="doe_grating_period",
            description="Diffractive optical element grating spacing (controls diffraction pattern)",
            min_value=0.1,
            max_value=2.0,
            current_value=1.0,
            step_size=0.2,
        ),
        OpticalVariable(
            name="surface_curvature",
            description="Lens surface curvature (controls focus depth and PSF spread)",
            min_value=0.0,
            max_value=1.0,
            current_value=0.5,
            step_size=0.1,
        ),
        OpticalVariable(
            name="chromatic_shift",
            description="Wavelength-dependent PSF centroid shift magnitude",
            min_value=0.0,
            max_value=1.0,
            current_value=0.3,
            step_size=0.1,
        ),
        OpticalVariable(
            name="depth_variation",
            description="Depth-dependent PSF variation magnitude across depth planes",
            min_value=0.0,
            max_value=1.0,
            current_value=0.5,
            step_size=0.1,
        ),
    ]


def build_default_optimization_spec(
    target_metrics: list[str] | None = None,
    backend: str = "mock_deeplens",
    objective: str = "",
) -> OptimizationSpec:
    """Build a default co-design optimization spec."""
    metrics = target_metrics or ["PSNR", "reconstruction_score"]
    return OptimizationSpec(
        optimization_id=make_deterministic_id("opt", objective or "codesign", backend),
        objective=objective,
        target_metrics=metrics,
        optical_variables=build_default_optical_variables(),
        loss_terms=[{"metric": m, "weight": 1.0, "direction": "maximize"} for m in metrics],
        variables=[],
        constraints={},
        max_iterations=20,
        budget={"max_seconds": 600, "max_iterations": 20},
        backend=backend,
        requires_native_support=False,
        metadata={"draft": False, "codesign": True},
    )
