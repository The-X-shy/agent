"""Schema tests for component surrogate PSF specs and results."""

import json

import pytest

from optiresearch.schemas.component_surrogate_psf import (
    ComponentSurrogateHSICoDesignResult,
    ComponentSurrogateHSICoDesignSpec,
    ComponentSurrogatePSFResult,
    ComponentSurrogatePSFSpec,
    make_component_surrogate_hsi_run_id,
)


def test_component_surrogate_psf_spec_defaults():
    spec = ComponentSurrogatePSFSpec(component_type="fresnel", psf_size=9, band_count=4)

    assert spec.normalize_psf is True
    assert spec.device == "cpu"
    assert spec.parameter_init == "default"
    assert spec.surrogate_model == "hybrid_simple"
    assert spec.strict_native_component is True
    assert spec.allow_proxy_psf is True


@pytest.mark.parametrize("component", ["fresnel", "binary2phase", "diffractive_candidate"])
def test_component_surrogate_psf_spec_accepts_supported_components(component):
    spec = ComponentSurrogatePSFSpec(component_type=component, psf_size=9, band_count=4)
    assert spec.component_type == component


def test_component_surrogate_psf_spec_rejects_unknown_component():
    with pytest.raises(Exception):
        ComponentSurrogatePSFSpec(component_type="geolens", psf_size=9, band_count=4)


def test_component_surrogate_psf_result_serializes_without_tensor_payload():
    result = ComponentSurrogatePSFResult(
        component_type="binary2phase",
        status="succeeded",
        evidence_level="component_surrogate_hsi_codesign",
        claim_ceiling="component_surrogate_hsi_codesign",
        psf_shape=[4, 9, 9],
        psf_requires_grad=True,
        parameter_count=7,
        trainable_param_count=7,
        params_with_grad=7,
        grad_norm_max=0.1,
        component_parameter_changed=False,
        psf_energy=[1.0, 1.0, 1.0, 1.0],
        psf_centroid=[4.0, 4.0],
        psf_width=1.5,
    )

    data = json.loads(json.dumps(result.model_dump(mode="json")))
    assert data["psf_requires_grad"] is True
    assert data["claim_ceiling"] == "component_surrogate_hsi_codesign"
    assert "psf" not in data
    assert "component_parameters" not in data


def test_component_surrogate_hsi_spec_defaults():
    spec = ComponentSurrogateHSICoDesignSpec(component_type="fresnel")

    assert spec.dataset == "synthetic"
    assert spec.steps == 3
    assert spec.device == "cpu"
    assert spec.psf_spec.component_type == "fresnel"
    assert spec.psf_spec.band_count == spec.band_count


def test_component_surrogate_hsi_result_serializes_core_metrics():
    result = ComponentSurrogateHSICoDesignResult(
        run_id="run1",
        component_type="fresnel",
        status="succeeded",
        reconstruction_loss_before=1.0,
        reconstruction_loss_after=0.9,
        mse_before=1.0,
        mse_after=0.9,
        psnr_before=10.0,
        psnr_after=11.0,
        sam_before=0.5,
        sam_after=0.4,
        component_grad_norm_max=0.2,
        component_parameter_changed=True,
        psf_requires_grad=True,
        loss_requires_grad=True,
        evidence_level="component_surrogate_hsi_codesign",
        claim_ceiling="component_surrogate_hsi_codesign",
    )

    data = result.model_dump(mode="json")
    assert data["component_parameter_changed"] is True
    assert data["loss_requires_grad"] is True


def test_make_component_surrogate_hsi_run_id_is_deterministic():
    first = make_component_surrogate_hsi_run_id("fresnel", "synthetic", 3)
    second = make_component_surrogate_hsi_run_id("fresnel", "synthetic", 3)
    assert first == second
    assert first.startswith("comp_sur_hsi_")
