"""Native GeoLens Stability Benchmark schema.

Multi-seed, multi-config reproducibility benchmark for stabilized
native GeoLens HSI optimization.
"""

from __future__ import annotations

from typing import Optional

from pydantic import Field

from optiresearch.memory.schemas import StrictModel, make_deterministic_id

BENCHMARK_SCHEMA_VERSION = "0.1"


class NativeGeoLensBenchmarkSpec(StrictModel):
    schema_version: str = Field(default=BENCHMARK_SCHEMA_VERSION)
    benchmark_id: Optional[str] = None
    lens_file: str = Field(default="auto:cooke")
    dataset: str = Field(default="synthetic")
    seeds: list[int] = Field(default_factory=lambda: [0, 1, 2])
    step_grid: list[int] = Field(default_factory=lambda: [10, 20])
    spectral_angle_weights: list[float] = Field(default_factory=lambda: [0.1, 0.2, 0.5])
    grad_clip_norms: list[float] = Field(default_factory=lambda: [1000.0])
    device: str = Field(default="cpu")
    max_configs: Optional[int] = None
    timeout_sec: Optional[int] = None
    save_artifacts: bool = True


class NativeGeoLensBenchmarkConfigResult(StrictModel):
    config_id: str
    seed: int
    steps: int
    spectral_angle_weight: float
    grad_clip_norm: float
    status: str
    evidence_level: Optional[str] = None
    error_code: Optional[str] = None
    parameter_count: int = 0
    trainable_param_count: int = 0
    graph_connected: bool = False
    psf_requires_grad: bool = False
    loss_requires_grad: bool = False
    parameter_changed: bool = False
    accepted_update_count: int = 0
    rollback_count: int = 0
    rollback_reasons: list[str] = Field(default_factory=list)
    mse_before: Optional[float] = None
    mse_after: Optional[float] = None
    mse_delta: Optional[float] = None
    mse_improved: bool = False
    psnr_before: Optional[float] = None
    psnr_after: Optional[float] = None
    psnr_delta: Optional[float] = None
    psnr_improved: bool = False
    sam_before: Optional[float] = None
    sam_after: Optional[float] = None
    sam_delta: Optional[float] = None
    sam_improved: bool = False
    grad_norm_max: Optional[float] = None
    grad_norm_mean: Optional[float] = None
    psf_centroid_shift: Optional[float] = None
    psf_width_shift: Optional[float] = None
    stability_score: Optional[float] = None
    metric_tradeoff_summary: str = ""
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class NativeGeoLensBenchmarkSummary(StrictModel):
    benchmark_id: str
    config_count: int = 0
    completed_count: int = 0
    unsupported_count: int = 0
    failed_count: int = 0
    completion_rate: float = 0.0
    seed_count: int = 0
    all_metrics_improved_count: int = 0
    all_metrics_improved_rate: float = 0.0
    all_metrics_improved_rate_full_grid: float = 0.0
    mse_improved_rate: float = 0.0
    psnr_improved_rate: float = 0.0
    sam_improved_rate: float = 0.0
    mean_mse_delta: Optional[float] = None
    std_mse_delta: Optional[float] = None
    mean_psnr_delta: Optional[float] = None
    std_psnr_delta: Optional[float] = None
    mean_sam_delta: Optional[float] = None
    std_sam_delta: Optional[float] = None
    mean_grad_norm_max: Optional[float] = None
    rollback_rate: float = 0.0
    best_config_id: str = ""
    robust_config_family: str = ""
    claim_recommendation: str = ""
    safe_wording: str = ""
    blocked_claims: list[str] = Field(default_factory=list)
    config_results: list[NativeGeoLensBenchmarkConfigResult] = Field(default_factory=list)


def make_benchmark_id() -> str:
    return make_deterministic_id("ngeo_bench", str(__import__("time").time()))
