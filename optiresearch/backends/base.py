"""Core OpticalBackend dataclass for the backend registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

BackendType = Literal["mock", "deeplens", "proxy", "synthetic", "external"]

DifferentiabilityLevel = Literal[
    "none",
    "black_box",
    "differentiable_proxy",
    "native_component",
    "native_lens_simulation",
    "native_waveoptics",
]


@dataclass(frozen=True)
class OpticalBackend:
    """Immutable description of an optical simulation/optimization backend.

    Each backend declares its capabilities and claim ceiling so that
    experiment controllers, strategy engines, and claim gates can make
    informed decisions without hardcoding backend-specific knowledge.
    """

    backend_id: str
    label: str
    backend_type: BackendType
    differentiability_level: DifferentiabilityLevel

    supports_psf_generation: bool = False
    supports_image_simulation: bool = False
    supports_hsi_forward: bool = False
    supports_native_optimization: bool = False
    supports_full_waveoptics: bool = False
    requires_lens_file: bool = False
    supports_remote_execution: bool = False
    supports_real_hardware: bool = False

    claim_ceiling: str = "unsupported"
    known_failure_modes: list[str] = field(default_factory=list)
    recommended_use_cases: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
