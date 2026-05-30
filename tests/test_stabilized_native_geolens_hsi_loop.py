"""Tests for stabilized native GeoLens HSI loop (Phase 65)."""

from __future__ import annotations

import json
import pytest

from optiresearch.schemas.native_geolens_stability import (
    NativeGeoLensStabilityResult,
    NativeGeoLensStabilitySpec,
)
from optiresearch.schemas.stable_native_lens_hsi import make_stable_lens_id


def _make_spec(**kwargs):
    defaults = {
        "run_id": make_stable_lens_id("GeoLensCooke", "differentiable_linear"),
        "candidate": "GeoLensCooke",
        "reconstructor": "differentiable_linear",
        "max_steps": 3,
        "optical_warmup_steps": 1,
        "save_artifacts": False,
    }
    defaults.update(kwargs)
    return NativeGeoLensStabilitySpec(**defaults)


def test_stabilized_loop_runs_without_crash():
    """Loop should run and return a structured result even if DeepLens unavailable."""
    spec = _make_spec()
    from optiresearch.runtime.stable_native_lens_hsi_loop import (
        run_stabilized_native_geolens_hsi_loop,
    )
    result = run_stabilized_native_geolens_hsi_loop(spec)
    assert isinstance(result, NativeGeoLensStabilityResult)
    assert result.run_id == spec.run_id
    assert result.status in ("succeeded", "unsupported", "failed")


def test_stabilized_result_includes_stability_fields():
    spec = _make_spec()
    from optiresearch.runtime.stable_native_lens_hsi_loop import (
        run_stabilized_native_geolens_hsi_loop,
    )
    result = run_stabilized_native_geolens_hsi_loop(spec)
    # Stability-specific fields always present
    assert isinstance(result.rollback_reasons, list)
    assert isinstance(result.warnings, list)
    assert isinstance(result.loss_terms_final, dict)
    assert isinstance(result.metric_tradeoff_summary, str)


def test_stabilized_result_inherits_base_fields():
    spec = _make_spec()
    from optiresearch.runtime.stable_native_lens_hsi_loop import (
        run_stabilized_native_geolens_hsi_loop,
    )
    result = run_stabilized_native_geolens_hsi_loop(spec)
    assert isinstance(result.parameter_count, int)
    assert isinstance(result.trainable_param_count, int)
    assert isinstance(result.graph_connected, bool)
    assert isinstance(result.psf_requires_grad, bool)


def test_spec_with_different_spectral_weights():
    for w in [0.05, 0.2, 0.5]:
        spec = _make_spec(spectral_angle_weight=w)
        assert spec.spectral_angle_weight == w


def test_spec_with_rollback_disabled():
    spec = _make_spec(enable_rollback_policy=False)
    assert spec.enable_rollback_policy is False


def test_result_serializable():
    spec = _make_spec()
    from optiresearch.runtime.stable_native_lens_hsi_loop import (
        run_stabilized_native_geolens_hsi_loop,
    )
    result = run_stabilized_native_geolens_hsi_loop(spec)
    data = result.model_dump(mode="json")
    assert isinstance(json.dumps(data, default=str), str)
