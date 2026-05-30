"""Tests for native_geolens_stability schema."""

from __future__ import annotations

import pytest

from optiresearch.schemas.native_geolens_stability import (
    NativeGeoLensStabilityResult,
    NativeGeoLensStabilitySpec,
)
from optiresearch.schemas.stable_native_lens_hsi import (
    StableNativeLensHSIResult,
    StableNativeLensHSISpec,
    make_stable_lens_id,
)


def test_spec_defaults():
    spec = NativeGeoLensStabilitySpec(
        run_id=make_stable_lens_id("GeoLensCooke", "differentiable_linear"),
        candidate="GeoLensCooke",
        reconstructor="differentiable_linear",
    )
    assert spec.spectral_angle_weight == 0.2
    assert spec.seed == 42
    assert spec.optimizer_name == "adam"
    assert spec.enable_rollback_policy is True
    assert spec.rollback_max_grad_norm == 5000.0


def test_spec_validation():
    spec = NativeGeoLensStabilitySpec(
        run_id=make_stable_lens_id("GeoLensCooke", "differentiable_linear"),
        candidate="GeoLensCooke",
        reconstructor="differentiable_linear",
        seed=0,
        spectral_angle_weight=5.0,
        rollback_max_grad_norm=1000.0,
    )
    assert spec.seed == 0
    assert spec.spectral_angle_weight == 5.0
    assert spec.rollback_max_grad_norm == 1000.0


def test_spec_rejects_invalid_optimizer():
    with pytest.raises(ValueError, match="optimizer_name"):
        NativeGeoLensStabilitySpec(
            run_id=make_stable_lens_id("GeoLensCooke", "differentiable_linear"),
            candidate="GeoLensCooke",
            reconstructor="differentiable_linear",
            optimizer_name="rmsprop",
        )


def test_result_defaults():
    result = NativeGeoLensStabilityResult(
        run_id="test",
        status="succeeded",
        candidate="GeoLensCooke",
        reconstructor="differentiable_linear",
    )
    assert result.spectral_angle_weight == 0.2
    assert result.rollback_reasons == []
    assert result.loss_terms_final == {}
    assert result.warnings == []
    assert result.metric_tradeoff_summary == ""


def test_result_inherits_stable_lens_fields():
    result = NativeGeoLensStabilityResult(
        run_id="test",
        status="succeeded",
        candidate="GeoLensCooke",
        reconstructor="differentiable_linear",
        trainable_param_count=14,
        parameter_count=14,
        graph_connected=True,
    )
    assert result.trainable_param_count == 14
    assert result.parameter_count == 14
    assert result.graph_connected is True
    assert isinstance(result, StableNativeLensHSIResult)


def test_result_roundtrip():
    result = NativeGeoLensStabilityResult(
        run_id="test",
        status="succeeded",
        candidate="GeoLensCooke",
        reconstructor="differentiable_linear",
        mse_before=0.5, mse_after=0.4,
        psnr_before=3.0, psnr_after=4.0,
        sam_before=1.0, sam_after=0.9,
        accepted_update_count=3,
        rollback_count=1,
        rollback_reasons=["mse_worse"],
        stability_score=0.85,
    )
    data = result.model_dump(mode="json")
    reloaded = NativeGeoLensStabilityResult.model_validate(data)
    assert reloaded.mse_before == 0.5
    assert reloaded.accepted_update_count == 3
    assert reloaded.rollback_reasons == ["mse_worse"]
    assert reloaded.stability_score == 0.85


def test_spec_is_stable_lens_spec():
    spec = NativeGeoLensStabilitySpec(
        run_id=make_stable_lens_id("GeoLensCooke", "differentiable_linear"),
        candidate="GeoLensCooke",
        reconstructor="differentiable_linear",
    )
    assert isinstance(spec, StableNativeLensHSISpec)
    # inherited fields
    assert spec.max_steps == 10
    assert spec.optical_lr == 1e-6
