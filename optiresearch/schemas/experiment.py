"""Standard experiment specification schemas."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import Field

from optiresearch.memory.schemas import StrictModel, make_deterministic_id


EXPERIMENT_SPEC_SCHEMA_VERSION = "0.1"


class OpticalSpec(StrictModel):
    schema_version: str = Field(default=EXPERIMENT_SPEC_SCHEMA_VERSION)
    spec_id: str = Field(min_length=1)
    encoder_type: Literal[
        "conventional",
        "achromatic",
        "edof",
        "chromatic_coded",
        "controlled_chromatic_edof",
        "mock",
    ]
    aperture: Optional[float]
    focal_length: Optional[float]
    f_number: Optional[float]
    sensor_type: Literal["mono", "rgb", "hsi", "mock"]
    wavelength_range_nm: tuple[float, float]
    wavelength_bands: int = Field(ge=1)
    depth_range_mm: tuple[float, float]
    depth_planes: int = Field(ge=1)
    psf_size: int = Field(ge=1)
    constraints: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SweepSpec(StrictModel):
    schema_version: str = Field(default=EXPERIMENT_SPEC_SCHEMA_VERSION)
    sweep_id: str = Field(min_length=1)
    wavelengths_nm: list[float] = Field(default_factory=list)
    depths_mm: list[float] = Field(default_factory=list)
    fields: list[float] = Field(default_factory=list)
    seeds: list[int] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MetricSpec(StrictModel):
    schema_version: str = Field(default=EXPERIMENT_SPEC_SCHEMA_VERSION)
    metric_id: str = Field(min_length=1)
    optical_metrics: list[str] = Field(default_factory=list)
    reconstruction_metrics: list[str] = Field(default_factory=list)
    evidence_metrics: list[str] = Field(default_factory=list)
    primary_metric: str = Field(min_length=1)
    maximize: bool
    thresholds: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExperimentSpec(StrictModel):
    schema_version: str = Field(default=EXPERIMENT_SPEC_SCHEMA_VERSION)
    experiment_id: str = Field(min_length=1)
    objective: str = Field(min_length=1)
    optical_spec: OpticalSpec
    sweep_spec: SweepSpec
    metric_spec: MetricSpec
    backend: Literal["mock_deeplens", "deeplens", "metasurface_mock"]
    run_budget: dict[str, Any] = Field(default_factory=dict)
    created_by: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


def build_default_mock_edof_hsi_experiment(
    objective: str,
    encoder_type: Literal[
        "conventional",
        "achromatic",
        "edof",
        "chromatic_coded",
        "controlled_chromatic_edof",
        "mock",
    ] = "controlled_chromatic_edof",
) -> ExperimentSpec:
    """Build a deterministic default experiment spec for the mock EDOF-HSI task."""

    wavelengths = _linspace(450.0, 700.0, 31)
    depths = _linspace(-4.0, 4.0, 9)
    optical = OpticalSpec(
        spec_id=make_deterministic_id("optical", objective, "mock_edof_hsi"),
        encoder_type=encoder_type,
        aperture=None,
        focal_length=None,
        f_number=2.8,
        sensor_type="hsi",
        wavelength_range_nm=(450.0, 700.0),
        wavelength_bands=31,
        depth_range_mm=(-4.0, 4.0),
        depth_planes=9,
        psf_size=32,
        constraints={
            "depth_invariant_target": True,
            "spectral_discriminative_target": True,
        },
        metadata={"backend": "mock_deeplens", "encoder_type": encoder_type},
    )
    sweep = SweepSpec(
        sweep_id=make_deterministic_id("sweep", objective, wavelengths, depths, 42),
        wavelengths_nm=wavelengths,
        depths_mm=depths,
        fields=[0.0],
        seeds=[42],
        metadata={"sampling": "uniform"},
    )
    metrics = MetricSpec(
        metric_id=make_deterministic_id("metric", objective, "mock_metrics"),
        optical_metrics=[
            "psf_depth_similarity",
            "spectral_separability",
            "mock_mtf_mean",
            "mock_energy_efficiency",
        ],
        reconstruction_metrics=[],
        evidence_metrics=["artifact_support", "trace_support"],
        primary_metric="psf_depth_similarity",
        maximize=True,
        thresholds={"psf_depth_similarity": 0.8, "spectral_separability": 0.3},
        metadata={"evidence_policy": "simulation_only"},
    )
    return ExperimentSpec(
        experiment_id=make_deterministic_id("experiment", objective, optical.spec_id, sweep.sweep_id),
        objective=objective,
        optical_spec=optical,
        sweep_spec=sweep,
        metric_spec=metrics,
        backend="mock_deeplens",
        run_budget={"max_runs": 1, "max_seconds": 60},
        created_by="MethodBuilder",
        metadata={"kind": "default_mock_edof_hsi", "encoder_type": encoder_type},
    )


def validate_experiment_spec_version(spec: ExperimentSpec) -> bool:
    """Validate that an experiment and all nested specs use frozen v0.1."""

    checks = {
        "ExperimentSpec schema_version": spec.schema_version,
        "OpticalSpec schema_version": spec.optical_spec.schema_version,
        "SweepSpec schema_version": spec.sweep_spec.schema_version,
        "MetricSpec schema_version": spec.metric_spec.schema_version,
    }
    for label, version in checks.items():
        if version != EXPERIMENT_SPEC_SCHEMA_VERSION:
            raise ValueError(f"{label} must be {EXPERIMENT_SPEC_SCHEMA_VERSION}, got {version}")
    return True


def _linspace(start: float, stop: float, count: int) -> list[float]:
    if count == 1:
        return [float(start)]
    step = (stop - start) / float(count - 1)
    return [round(start + idx * step, 6) for idx in range(count)]
