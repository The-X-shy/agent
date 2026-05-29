"""Gradient tests for component surrogate PSF kernels."""

import pytest

from optiresearch.optics.component_surrogate_psf import build_component_surrogate_psf
from optiresearch.schemas.component_surrogate_psf import ComponentSurrogatePSFSpec


@pytest.mark.parametrize("component", ["fresnel", "binary2phase"])
def test_psf_loss_backpropagates_to_component_parameters(component):
    result = build_component_surrogate_psf(
        ComponentSurrogatePSFSpec(component_type=component, psf_size=9, band_count=4)
    )

    params = result.component_parameters
    for param in params:
        if param.grad is not None:
            param.grad.zero_()
    loss = result.psf.pow(2).sum()
    loss.backward()

    grads = [p.grad for p in params if p.grad is not None and p.grad.abs().sum() > 0]
    assert result.psf.requires_grad is True
    assert loss.requires_grad is True
    assert len(grads) == len(params)


@pytest.mark.parametrize("component", ["fresnel", "binary2phase"])
def test_optimizer_step_changes_component_parameters(component):
    result = build_component_surrogate_psf(
        ComponentSurrogatePSFSpec(component_type=component, psf_size=9, band_count=4)
    )
    params = result.component_parameters
    before = [p.detach().clone() for p in params]
    optimizer = __import__("torch").optim.SGD(params, lr=0.1)

    optimizer.zero_grad()
    loss = result.psf[:, 4, 4].sum()
    loss.backward()
    optimizer.step()

    changed = [not p.detach().allclose(b) for p, b in zip(params, before)]
    assert any(changed)
