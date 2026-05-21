"""Tests for differentiable HSI reconstructor."""

import torch
from optiresearch.hsi.differentiable_reconstructor import (
    DifferentiableLinearHSIReconstructor,
    TinyDifferentiableHSIReconstructor,
    build_psf_condition_features,
    hsi_reconstruction_losses,
)


def test_build_psf_features_shape():
    psf = torch.randn(8, 16, 16)
    feats = build_psf_condition_features(psf)
    assert feats.shape == (8, 4)


def test_build_psf_features_preserves_grad():
    psf_param = torch.randn(4, 8, 8, requires_grad=True)
    psf = psf_param.pow(2)
    feats = build_psf_condition_features(psf)
    loss = feats.sum()
    loss.backward()
    assert psf_param.grad is not None


def test_linear_reconstructor_output_shape():
    recon = DifferentiableLinearHSIReconstructor(bands=8)
    measurement = torch.randn(2, 1, 16, 16)
    psf = torch.randn(8, 5, 5)
    output = recon(measurement, psf)
    assert output.shape == (2, 8, 16, 16)


def test_tiny_cnn_reconstructor_output_shape():
    recon = TinyDifferentiableHSIReconstructor(bands=8)
    measurement = torch.randn(2, 1, 16, 16)
    psf = torch.randn(8, 5, 5)
    output = recon(measurement, psf)
    assert output.shape == (2, 8, 16, 16)


def test_reconstructor_params_have_grad_after_backward():
    recon = DifferentiableLinearHSIReconstructor(bands=4)
    measurement = torch.randn(1, 1, 8, 8)
    psf = torch.randn(4, 5, 5)
    target = torch.randn(1, 4, 8, 8)

    output = recon(measurement, psf)
    losses = hsi_reconstruction_losses(output, target)
    losses["total_loss"].backward()

    for name, param in recon.named_parameters():
        assert param.grad is not None, f"{name} has no grad"


def test_psf_grad_flows_through_reconstructor():
    recon = DifferentiableLinearHSIReconstructor(bands=4)
    psf_param = torch.randn(4, 5, 5, requires_grad=True)
    psf = psf_param.pow(2)
    measurement = torch.randn(1, 1, 8, 8)
    target = torch.randn(1, 4, 8, 8)

    output = recon(measurement, psf)
    losses = hsi_reconstruction_losses(output, target, measurement, psf)
    losses["total_loss"].backward()

    assert psf_param.grad is not None
    assert psf_param.grad.abs().sum() > 0


def test_hsi_reconstruction_losses_requires_grad():
    recon = torch.randn(2, 4, 16, 16, requires_grad=True)
    target = torch.randn(2, 4, 16, 16)
    losses = hsi_reconstruction_losses(recon, target)
    assert losses["total_loss"].requires_grad is True
    assert losses["mse_loss"].requires_grad is True


def test_linear_reconstructor_is_nn_module():
    recon = DifferentiableLinearHSIReconstructor(bands=8)
    assert isinstance(recon, torch.nn.Module)
    params = list(recon.parameters())
    assert len(params) > 0


def test_tiny_cnn_reconstructor_is_nn_module():
    recon = TinyDifferentiableHSIReconstructor(bands=8)
    assert isinstance(recon, torch.nn.Module)
    params = list(recon.parameters())
    assert len(params) > 0
