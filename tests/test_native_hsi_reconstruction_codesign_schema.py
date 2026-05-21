"""Tests for Native HSI Reconstruction CoDesign schemas."""

import pytest
from optiresearch.schemas.native_hsi_reconstruction_codesign import (
    NativeHSIReconstructionCoDesignResult,
    NativeHSIReconstructionCoDesignSpec,
    make_recon_codesign_id,
)


def test_spec_creates_with_valid_component_and_reconstructor():
    spec = NativeHSIReconstructionCoDesignSpec(
        run_id=make_recon_codesign_id("Fresnel", "differentiable_linear"),
        optical_component="Fresnel",
        reconstructor="differentiable_linear",
    )
    assert spec.optical_component == "Fresnel"
    assert spec.reconstructor == "differentiable_linear"
    assert spec.bands == 31
    assert spec.max_steps == 5
    assert spec.optimize_optics is True
    assert spec.optimize_reconstructor is True


def test_spec_rejects_invalid_component():
    with pytest.raises(ValueError, match="optical_component"):
        NativeHSIReconstructionCoDesignSpec(
            run_id="test",
            optical_component="BadComponent",
            reconstructor="differentiable_linear",
        )


def test_spec_rejects_invalid_reconstructor():
    with pytest.raises(ValueError, match="reconstructor"):
        NativeHSIReconstructionCoDesignSpec(
            run_id="test",
            optical_component="Fresnel",
            reconstructor="bad_reconstructor",
        )


def test_spec_rejects_invalid_device():
    with pytest.raises(ValueError, match="device"):
        NativeHSIReconstructionCoDesignSpec(
            run_id="test",
            optical_component="Fresnel",
            reconstructor="differentiable_linear",
            device="tpu",
        )


def test_result_defaults():
    result = NativeHSIReconstructionCoDesignResult(
        run_id="test",
        status="unsupported",
        optical_component="Fresnel",
    )
    assert result.differentiable is False
    assert result.full_reconstruction_loss_used is False
    assert result.full_wave_optics is False
    assert result.phase_to_fft_proxy_used is True
    assert result.optimizer_step_executed is False


def test_result_succeeded_has_all_fields():
    result = NativeHSIReconstructionCoDesignResult(
        run_id="test",
        status="succeeded",
        optical_component="Fresnel",
        reconstructor="differentiable_linear",
        differentiable=True,
        full_reconstruction_loss_used=True,
        reconstruction_loss_before=0.5,
        reconstruction_loss_after=0.3,
        optical_gradient_norm=0.01,
        recon_gradient_norm=0.05,
        optical_parameters_changed=True,
        optimizer_step_executed=True,
        evidence_level="native_full_reconstruction_proxy",
    )
    assert result.status == "succeeded"
    assert result.differentiable is True
    assert result.evidence_level == "native_full_reconstruction_proxy"


def test_make_recon_codesign_id_deterministic():
    a = make_recon_codesign_id("Fresnel", "differentiable_linear")
    b = make_recon_codesign_id("Fresnel", "differentiable_linear")
    assert a == b
    assert a.startswith("recon_codesign_")
