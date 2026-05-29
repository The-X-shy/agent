"""Tests for differentiable component surrogate PSF builder."""

import pytest
import torch

from optiresearch.optics.component_surrogate_psf import build_component_surrogate_psf
from optiresearch.schemas.component_surrogate_psf import ComponentSurrogatePSFSpec


@pytest.mark.parametrize(
    ("component", "expected_params"),
    [("fresnel", 1), ("binary2phase", 7)],
)
def test_build_component_surrogate_psf_succeeds(component, expected_params):
    spec = ComponentSurrogatePSFSpec(component_type=component, psf_size=9, band_count=4)

    result = build_component_surrogate_psf(spec)

    assert result.status == "succeeded"
    assert result.psf_shape == [4, 9, 9]
    assert result.psf_requires_grad is True
    assert result.parameter_count == expected_params
    assert result.trainable_param_count == expected_params
    assert result.params_with_grad > 0
    assert result.grad_norm_max > 0
    assert torch.is_tensor(result.psf)
    assert result.psf.requires_grad is True
    assert result.claim_ceiling == "component_surrogate_hsi_codesign"
    assert all(abs(float(v) - 1.0) < 1e-5 for v in result.psf_energy)


def test_fresnel_surrogate_psf_changes_with_component_parameter():
    spec = ComponentSurrogatePSFSpec(component_type="fresnel", psf_size=9, band_count=4)
    first = build_component_surrogate_psf(spec)
    params = first.component_parameters
    params[0].data.add_(5.0)
    second = build_component_surrogate_psf(spec, initial_parameters=params)

    assert not torch.allclose(first.psf.detach(), second.psf.detach())


def test_binary2phase_uses_polynomial_phase_parameters():
    spec = ComponentSurrogatePSFSpec(component_type="binary2phase", psf_size=9, band_count=4)
    result = build_component_surrogate_psf(spec)

    assert result.parameter_names == ["d", "order2", "order4", "order6", "order8", "order10", "order12"]
    assert result.params_with_grad == 7


def test_diffractive_candidate_returns_structured_surrogate_or_followup():
    spec = ComponentSurrogatePSFSpec(component_type="diffractive_candidate", psf_size=9, band_count=4)

    result = build_component_surrogate_psf(spec)

    assert result.status in ("succeeded", "needs_followup")
    assert result.evidence_level in ("component_surrogate_hsi_codesign", "diagnostic_evidence")
    if result.status == "succeeded":
        assert result.psf_requires_grad is True


def test_disconnected_psf_reports_error(monkeypatch):
    from optiresearch.optics import component_surrogate_psf as builder

    def _bad_fresnel(*_args, **_kwargs):
        psf = torch.ones(4, 9, 9)
        psf = psf / psf.sum(dim=(1, 2), keepdim=True)
        return psf, [torch.nn.Parameter(torch.tensor(1.0))], ["f0"]

    monkeypatch.setattr(builder, "_build_fresnel_psf", _bad_fresnel)
    spec = ComponentSurrogatePSFSpec(component_type="fresnel", psf_size=9, band_count=4)

    result = build_component_surrogate_psf(spec)

    assert result.status == "failed"
    assert result.error_code == "PSF_GRAPH_DISCONNECTED"
    assert result.psf_requires_grad is False
