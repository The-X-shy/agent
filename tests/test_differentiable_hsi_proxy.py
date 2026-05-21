"""Tests for differentiable HSI proxy loss."""

import torch
from optiresearch.hsi.differentiable_proxy import (
    generate_torch_synthetic_hsi,
    make_measurement_from_psf_torch,
    reconstruct_proxy_torch,
    hsi_proxy_loss,
)


def test_generate_synthetic_hsi_shape():
    hsi = generate_torch_synthetic_hsi(batch=2, bands=31, height=32, width=32)
    assert hsi.shape == (2, 31, 32, 32)
    assert hsi.dtype == torch.float32


def test_make_measurement_shape():
    hsi = generate_torch_synthetic_hsi(batch=1, bands=5, height=16, width=16)
    psf = torch.randn(5, 7, 7)
    meas = make_measurement_from_psf_torch(hsi, psf)
    assert meas.shape == (1, 1, 16, 16)


def test_reconstruct_proxy_shape():
    meas = torch.randn(2, 1, 16, 16)
    psf = torch.randn(8, 5, 5)
    recon = reconstruct_proxy_torch(meas, psf, bands=8)
    assert recon.shape == (2, 8, 16, 16)


def test_loss_requires_grad_through_psf():
    """The critical test: loss must be differentiable w.r.t. PSF parameters."""
    hsi = generate_torch_synthetic_hsi(batch=1, bands=4, height=16, width=16)
    psf_param = torch.randn(4, 5, 5, requires_grad=True)
    psf = psf_param.pow(2)

    meas = make_measurement_from_psf_torch(hsi, psf)
    recon = reconstruct_proxy_torch(meas, psf, bands=4)
    loss = hsi_proxy_loss(recon, hsi, mode="mse")

    assert loss.requires_grad is True
    loss.backward()
    assert psf_param.grad is not None
    assert psf_param.grad.abs().sum() > 0


def test_loss_decreases_with_better_psf():
    """Loss should be lower when PSF is identity-like vs random."""
    hsi = generate_torch_synthetic_hsi(batch=1, bands=3, height=16, width=16)
    good_psf = torch.zeros(3, 5, 5)
    for b in range(3):
        good_psf[b, 2, 2] = 1.0
    bad_psf = torch.randn(3, 5, 5)

    meas_good = make_measurement_from_psf_torch(hsi, good_psf)
    recon_good = reconstruct_proxy_torch(meas_good, good_psf, bands=3)
    loss_good = hsi_proxy_loss(recon_good, hsi).item()

    meas_bad = make_measurement_from_psf_torch(hsi, bad_psf)
    recon_bad = reconstruct_proxy_torch(meas_bad, bad_psf, bands=3)
    loss_bad = hsi_proxy_loss(recon_bad, hsi).item()

    assert loss_good < loss_bad


def test_no_numpy_in_module():
    """Verify the module does not import numpy, detach, or no_grad."""
    import optiresearch.hsi.differentiable_proxy as mod
    source = open(mod.__file__).read()
    assert "import numpy" not in source
    assert "from numpy" not in source
    assert ".numpy()" not in source
    assert ".detach()" not in source
    assert "torch.no_grad()" not in source


def test_measurement_consistency_mode():
    hsi = generate_torch_synthetic_hsi(batch=1, bands=3, height=8, width=8)
    recon = torch.randn(1, 3, 8, 8, requires_grad=True)
    loss = hsi_proxy_loss(recon, hsi, mode="measurement_consistency")
    assert loss.requires_grad is True
