"""Tests for Native Optical-HSI CoDesign schemas."""

import pytest
from optiresearch.schemas.native_hsi_codesign import (
    NativeOpticalHSICoDesignResult,
    NativeOpticalHSICoDesignSpec,
    make_hsi_codesign_id,
    VALID_OPTICAL_COMPONENTS,
    VALID_HSI_OBJECTIVES,
)


def test_spec_creates_with_valid_component_and_objective():
    spec = NativeOpticalHSICoDesignSpec(
        run_id=make_hsi_codesign_id("Fresnel", "minimize_hsi_proxy_loss"),
        optical_component="Fresnel",
        objective="minimize_hsi_proxy_loss",
    )
    assert spec.optical_component == "Fresnel"
    assert spec.bands == 31
    assert spec.max_steps == 3
    assert spec.device == "cpu"
    assert spec.save_artifacts is True


def test_spec_rejects_invalid_component():
    with pytest.raises(ValueError, match="optical_component"):
        NativeOpticalHSICoDesignSpec(
            run_id="test",
            optical_component="InvalidComponent",
            objective="minimize_hsi_proxy_loss",
        )


def test_spec_rejects_invalid_objective():
    with pytest.raises(ValueError, match="objective"):
        NativeOpticalHSICoDesignSpec(
            run_id="test",
            optical_component="Fresnel",
            objective="bad_objective",
        )


def test_spec_rejects_invalid_device():
    with pytest.raises(ValueError, match="device"):
        NativeOpticalHSICoDesignSpec(
            run_id="test",
            optical_component="Fresnel",
            objective="minimize_hsi_proxy_loss",
            device="tpu",
        )


def test_result_defaults():
    result = NativeOpticalHSICoDesignResult(
        run_id="test",
        status="unsupported",
        optical_component="Fresnel",
    )
    assert result.differentiable is False
    assert result.optimizer_step_executed is False
    assert result.caveats == []
    assert result.autograd_break_detected is False


def test_result_succeeded_has_all_fields():
    result = NativeOpticalHSICoDesignResult(
        run_id="test",
        status="succeeded",
        optical_component="Fresnel",
        objective="minimize_hsi_proxy_loss",
        differentiable=True,
        gradient_norm=0.001,
        hsi_loss_before=0.1,
        hsi_loss_after=0.09,
        parameters_changed=True,
        optimizer_step_executed=True,
        evidence_level="native_hsi_proxy",
    )
    assert result.status == "succeeded"
    assert result.differentiable is True
    assert result.evidence_level == "native_hsi_proxy"


def test_make_hsi_codesign_id_is_deterministic():
    a = make_hsi_codesign_id("Fresnel", "minimize_hsi_proxy_loss")
    b = make_hsi_codesign_id("Fresnel", "minimize_hsi_proxy_loss")
    assert a == b
    assert a.startswith("native_hsi_codesign_")
