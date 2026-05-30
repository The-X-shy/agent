"""Tests for native_geolens_multiobjective_loss module."""

from __future__ import annotations

import pytest
import torch

from optiresearch.runtime.native_geolens_multiobjective_loss import (
    compute_native_geolens_multiobjective_loss,
    DEFAULT_WEIGHTS,
)


def _make_tensors(batch=1, bands=4, h=8, w=8, k=5, device="cpu"):
    recon = torch.rand(batch, bands, h, w, device=device)
    target = torch.rand(batch, bands, h, w, device=device)
    measurement = torch.rand(batch, h, w, device=device)
    psf = torch.rand(bands, k, k, device=device)
    psf_initial = torch.rand(bands, k, k, device=device)
    p = [torch.tensor(1.0, requires_grad=True, device=device)]
    p0 = [torch.tensor(1.0, device=device)]
    return recon, target, measurement, psf, psf_initial, p, p0


def test_all_loss_terms_computed():
    recon, target, meas, psf, psf_init, p, p0 = _make_tensors()
    terms = compute_native_geolens_multiobjective_loss(
        recon, target, meas, psf, psf_init, p, p0,
    )
    expected_keys = {
        "reconstruction_mse", "reconstruction_mae", "spectral_angle", "measurement_consistency",
        "psf_energy", "psf_centroid", "psf_width",
        "optical_param_delta", "smoothness", "total_loss",
    }
    assert set(terms.keys()) == expected_keys
    for k, v in terms.items():
        assert isinstance(v, torch.Tensor), f"{k} is not a tensor"
        if k != "total_loss":
            assert v.ndim == 0, f"{k} is not scalar"


def test_total_loss_requires_grad():
    recon, target, meas, psf, psf_init, p, p0 = _make_tensors()
    terms = compute_native_geolens_multiobjective_loss(
        recon, target, meas, psf, psf_init, p, p0,
    )
    assert terms["total_loss"].requires_grad


def test_grad_flows_to_optical_params():
    recon, target, meas, psf, psf_init, p, p0 = _make_tensors()
    # psf depends on p (simulate differentiable PSF)
    psf = psf.detach().clone()
    psf_param = torch.nn.Parameter(psf)
    psf_with_grad = psf_param * p[0]  # PSF scaled by optical param
    terms = compute_native_geolens_multiobjective_loss(
        recon, target, meas, psf_with_grad, psf_init, p, p0,
    )
    terms["total_loss"].backward()
    assert p[0].grad is not None
    assert p[0].grad.abs().max() > 0


def test_spectral_angle_is_real_acos():
    recon, target, meas, psf, psf_init, p, p0 = _make_tensors()
    terms = compute_native_geolens_multiobjective_loss(
        recon, target, meas, psf, psf_init, p, p0,
        weights={"reconstruction_mse": 0.0, "spectral_angle": 1.0,
                 "measurement_consistency": 0.0, "psf_energy": 0.0,
                 "psf_centroid": 0.0, "psf_width": 0.0,
                 "optical_param_delta": 0.0},
    )
    # SAM should be in [0, pi] for acos-based computation
    assert 0.0 <= float(terms["spectral_angle"].detach().cpu()) <= 3.2


def test_psf_energy_reg_zero_when_unchanged():
    _, _, _, psf, psf_init, p, p0 = _make_tensors()
    psf_same = psf_init.detach().clone()
    terms = compute_native_geolens_multiobjective_loss(
        torch.zeros(1, 4, 8, 8), torch.zeros(1, 4, 8, 8),
        None, psf_same, psf_init, p, p0,
    )
    assert float(terms["psf_energy"].detach().cpu()) == pytest.approx(0.0, abs=1e-6)


def test_psf_centroid_reg_zero_when_unchanged():
    _, _, _, _, psf_init, p, p0 = _make_tensors()
    psf_same = psf_init.detach().clone()
    terms = compute_native_geolens_multiobjective_loss(
        torch.zeros(1, 4, 8, 8), torch.zeros(1, 4, 8, 8),
        None, psf_same, psf_init, p, p0,
    )
    assert float(terms["psf_centroid"].detach().cpu()) == pytest.approx(0.0, abs=1e-6)


def test_psf_width_reg_zero_when_unchanged():
    _, _, _, _, psf_init, p, p0 = _make_tensors()
    psf_same = psf_init.detach().clone()
    terms = compute_native_geolens_multiobjective_loss(
        torch.zeros(1, 4, 8, 8), torch.zeros(1, 4, 8, 8),
        None, psf_same, psf_init, p, p0,
    )
    assert float(terms["psf_width"].detach().cpu()) == pytest.approx(0.0, abs=1e-6)


def test_custom_weights_override_defaults():
    recon, target, meas, psf, psf_init, p, p0 = _make_tensors()
    custom = {"reconstruction_mse": 10.0, "spectral_angle": 0.0,
              "measurement_consistency": 0.0, "psf_energy": 0.0,
              "psf_centroid": 0.0, "psf_width": 0.0,
              "reconstruction_mae": 0.0, "optical_param_delta": 0.0}
    terms = compute_native_geolens_multiobjective_loss(
        recon, target, meas, psf, psf_init, p, p0, weights=custom,
    )
    # With only MSE weight active, total_loss should be 10 * mse
    expected = 10.0 * terms["reconstruction_mse"]
    assert float(terms["total_loss"].detach().cpu()) == pytest.approx(
        float(expected.detach().cpu()), rel=1e-4,
    )


def test_measurement_consistency_zero_when_measurement_none():
    recon, target, _, psf, psf_init, p, p0 = _make_tensors()
    terms = compute_native_geolens_multiobjective_loss(
        recon, target, None, psf, psf_init, p, p0,
    )
    assert float(terms["measurement_consistency"].detach().cpu()) == 0.0


def test_default_weights_all_present():
    expected = {
        "reconstruction_mse", "reconstruction_mae", "spectral_angle",
        "measurement_consistency", "psf_energy", "psf_centroid",
        "psf_width", "optical_param_delta", "smoothness",
    }
    assert set(DEFAULT_WEIGHTS.keys()) == expected
