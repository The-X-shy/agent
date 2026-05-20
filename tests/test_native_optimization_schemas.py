"""Tests for native optimization probe schemas."""

from __future__ import annotations

import json

import pytest

from optiresearch.schemas.native_optimization import (
    NativeOptimizationProbeResult,
    NativeOptimizationProbeSpec,
    build_default_paraxial_psf_width_probe,
    make_probe_id,
)


def test_make_probe_id_is_deterministic():
    id1 = make_probe_id("ParaxialLens", "minimize_psf_width")
    id2 = make_probe_id("ParaxialLens", "minimize_psf_width")
    assert id1 == id2
    assert id1.startswith("native_opt_probe_")
    assert len(id1) > len("native_opt_probe_")


def test_make_probe_id_differs_by_lens_class():
    id1 = make_probe_id("ParaxialLens", "minimize_psf_width")
    id2 = make_probe_id("GeoLens", "minimize_psf_width")
    assert id1 != id2


def test_build_default_spec():
    spec = build_default_paraxial_psf_width_probe()
    assert spec.lens_class == "ParaxialLens"
    assert spec.objective == "minimize_psf_width"
    assert spec.max_steps == 2
    assert spec.strict_native is True
    assert spec.allow_adapter_proxy is False


def test_spec_rejects_invalid_lens_class():
    with pytest.raises(ValueError, match="lens_class must be one of"):
        NativeOptimizationProbeSpec(
            probe_id="test_1",
            lens_class="InvalidLens",
            objective="minimize_psf_width",
        )


def test_spec_rejects_invalid_objective():
    with pytest.raises(ValueError, match="objective must be one of"):
        NativeOptimizationProbeSpec(
            probe_id="test_2",
            lens_class="ParaxialLens",
            objective="bad_objective",
        )


def test_spec_rejects_invalid_device():
    with pytest.raises(ValueError, match="device must be"):
        NativeOptimizationProbeSpec(
            probe_id="test_3",
            lens_class="ParaxialLens",
            objective="minimize_psf_width",
            device="tpu",
        )


def test_spec_defaults():
    spec = NativeOptimizationProbeSpec(
        probe_id="test_4",
        lens_class="ParaxialLens",
        objective="minimize_psf_width",
    )
    assert spec.max_steps == 2
    assert spec.learning_rate == 1e-3
    assert spec.device == "cpu"
    assert spec.strict_native is True
    assert spec.allow_adapter_proxy is False
    assert spec.save_artifacts is True


def test_spec_serialization_round_trip():
    spec = NativeOptimizationProbeSpec(
        probe_id="test_5",
        lens_class="GeoLens",
        objective="maximize_center_intensity",
        max_steps=5,
        learning_rate=1e-2,
        device="cuda",
        strict_native=False,
        allow_adapter_proxy=True,
        save_artifacts=False,
    )
    payload = spec.model_dump()
    restored = NativeOptimizationProbeSpec(**payload)
    assert restored.lens_class == spec.lens_class
    assert restored.objective == spec.objective
    assert restored.max_steps == 5
    assert restored.learning_rate == 1e-2
    assert restored.device == "cuda"
    assert restored.strict_native is False
    assert restored.allow_adapter_proxy is True


def test_result_defaults():
    result = NativeOptimizationProbeResult(
        probe_id="test_6",
        status="succeeded",
        lens_class="ParaxialLens",
        realization_level="native",
    )
    assert result.differentiable is False
    assert result.native_parameter_update is False
    assert result.autograd_graph_exists is False
    assert result.loss_before is None
    assert result.loss_after is None
    assert result.artifact_paths == []
    assert result.caveats == []


def test_result_with_metrics():
    result = NativeOptimizationProbeResult(
        probe_id="test_7",
        status="succeeded",
        lens_class="ParaxialLens",
        objective="minimize_psf_width",
        realization_level="native",
        differentiable=True,
        native_parameter_update=True,
        autograd_graph_exists=True,
        loss_before=0.5,
        loss_after=0.3,
        parameter_norm_before=10.0,
        parameter_norm_after=9.8,
        gradient_norm=0.15,
        parameters_changed=True,
        optimizer_class="Adam",
    )
    payload = result.model_dump()
    assert payload["differentiable"] is True
    assert payload["native_parameter_update"] is True
    assert payload["gradient_norm"] == 0.15
    assert payload["parameters_changed"] is True


def test_result_serialization_round_trip():
    result = NativeOptimizationProbeResult(
        probe_id="test_8",
        status="unsupported",
        lens_class="PSFNetLens",
        objective="match_target_psf",
        realization_level="unavailable",
        error_code="DEEPLENS_NOT_INSTALLED",
        error_message="DeepLens package not found",
        caveats=["No DeepLens installation detected"],
    )
    payload = result.model_dump()
    restored = NativeOptimizationProbeResult(**payload)
    assert restored.status == "unsupported"
    assert restored.realization_level == "unavailable"
    assert restored.error_code == "DEEPLENS_NOT_INSTALLED"


def test_result_rejects_unknown_fields():
    with pytest.raises(ValueError):
        NativeOptimizationProbeResult(
            probe_id="test_9",
            status="succeeded",
            lens_class="ParaxialLens",
            realization_level="native",
            unknown_field=42,
        )


def test_spec_rejects_unknown_fields():
    with pytest.raises(ValueError):
        NativeOptimizationProbeSpec(
            probe_id="test_10",
            lens_class="ParaxialLens",
            objective="minimize_psf_width",
            bad_field=True,
        )


def test_all_valid_lens_classes_accepted():
    for cls_name in ["ParaxialLens", "GeoLens", "DiffractiveLens", "HybridLens", "PSFNetLens"]:
        spec = NativeOptimizationProbeSpec(
            probe_id=f"test_{cls_name}",
            lens_class=cls_name,
            objective="minimize_psf_width",
        )
        assert spec.lens_class == cls_name


def test_all_valid_objectives_accepted():
    for obj in [
        "minimize_psf_width",
        "maximize_center_intensity",
        "match_target_psf",
        "hsi_reconstruction_loss",
    ]:
        spec = NativeOptimizationProbeSpec(
            probe_id=f"test_{obj}",
            lens_class="ParaxialLens",
            objective=obj,
        )
        assert spec.objective == obj


def test_result_json_serializable():
    result = NativeOptimizationProbeResult(
        probe_id="test_json",
        status="succeeded",
        lens_class="ParaxialLens",
        objective="minimize_psf_width",
        realization_level="native",
        differentiable=True,
        native_parameter_update=True,
        gradient_norm=0.123,
        parameters_changed=True,
        loss_before=0.5,
        loss_after=0.3,
    )
    payload = result.model_dump()
    json_str = json.dumps(payload)
    restored = json.loads(json_str)
    assert restored["status"] == "succeeded"
    assert restored["gradient_norm"] == 0.123
