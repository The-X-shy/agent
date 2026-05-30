"""Tests for native_geolens_benchmark schema."""

from __future__ import annotations

from optiresearch.schemas.native_geolens_benchmark import (
    NativeGeoLensBenchmarkConfigResult,
    NativeGeoLensBenchmarkSpec,
    NativeGeoLensBenchmarkSummary,
    make_benchmark_id,
)


def test_spec_defaults():
    spec = NativeGeoLensBenchmarkSpec()
    assert spec.seeds == [0, 1, 2]
    assert spec.step_grid == [10, 20]
    assert spec.spectral_angle_weights == [0.1, 0.2, 0.5]
    assert spec.grad_clip_norms == [1000.0]


def test_config_result_fields():
    r = NativeGeoLensBenchmarkConfigResult(
        config_id="cfg_0",
        seed=0, steps=10,
        spectral_angle_weight=0.2, grad_clip_norm=1000,
        status="succeeded",
    )
    assert r.config_id == "cfg_0"
    assert r.seed == 0
    assert r.mse_improved is False
    assert r.rollback_reasons == []


def test_summary_defaults():
    s = NativeGeoLensBenchmarkSummary(benchmark_id="test")
    assert s.benchmark_id == "test"
    assert s.all_metrics_improved_rate == 0.0
    assert s.config_results == []


def test_make_benchmark_id():
    bid = make_benchmark_id()
    assert bid.startswith("ngeo_bench_")
    assert len(bid) > 12


def test_summary_roundtrip():
    config = NativeGeoLensBenchmarkConfigResult(
        config_id="cfg_0", seed=0, steps=10,
        spectral_angle_weight=0.2, grad_clip_norm=1000,
        status="succeeded", mse_before=0.5, mse_after=0.4,
        mse_improved=True, psnr_improved=True, sam_improved=True,
    )
    summary = NativeGeoLensBenchmarkSummary(
        benchmark_id="test",
        config_count=1, completed_count=1,
        all_metrics_improved_rate=1.0,
        config_results=[config],
    )
    data = summary.model_dump(mode="json")
    reloaded = NativeGeoLensBenchmarkSummary.model_validate(data)
    assert reloaded.all_metrics_improved_rate == 1.0
    assert len(reloaded.config_results) == 1
