"""Tests for optical objective library."""

import pytest
import torch
from optiresearch.objectives.hsi_objectives import (
    band_weighted_mse,
    measurement_consistency_loss,
    reconstruction_mse,
    spectral_angle_loss,
    spectral_smoothness_loss,
    task_aligned_hsi_loss,
)
from optiresearch.objectives.optical_objectives import (
    PRESET_PROFILES,
    ObjectiveProfile,
    field_consistency_loss,
    get_objective_profile,
    list_objective_profiles,
    psf_centroid_loss,
    psf_energy_loss,
    psf_smoothness_loss,
    psf_width_loss,
    spot_size_loss,
)
from optiresearch.objectives.regularizers import (
    optical_param_delta_limit,
    optical_param_l2,
    psf_centroid_preservation,
    psf_energy_preservation,
    psf_width_preservation,
    rollback_penalty,
)


# ── Optical losses ─────────────────────────────────────────────────

def test_psf_width_loss_is_differentiable():
    psf = torch.rand(1, 32, 32, requires_grad=True)
    loss = psf_width_loss(psf, target_width=2.0)
    loss.backward()
    assert psf.grad is not None
    assert psf.grad.abs().sum() > 0


def test_psf_centroid_loss_is_differentiable():
    psf = torch.rand(1, 32, 32, requires_grad=True)
    loss = psf_centroid_loss(psf)
    loss.backward()
    assert psf.grad is not None


def test_psf_energy_loss():
    psf = torch.ones(1, 8, 8)
    loss = psf_energy_loss(psf, target_energy=64.0)
    assert loss.item() == 0.0


def test_psf_smoothness_loss_nonnegative():
    psf = torch.rand(2, 16, 16)
    loss = psf_smoothness_loss(psf)
    assert loss.item() >= 0.0


def test_spot_size_loss():
    psf = torch.zeros(1, 16, 16)
    psf[0, 7, 7] = 1.0
    loss = spot_size_loss(psf, threshold=0.5)
    assert 0.0 < loss.item() < 1.0


def test_field_consistency_loss():
    psf_cube = torch.rand(4, 1, 32, 32)
    loss = field_consistency_loss(psf_cube)
    assert loss.item() >= 0.0


# ── HSI losses ──────────────────────────────────────────────────────

def test_reconstruction_mse_zero():
    t = torch.rand(2, 3, 16, 16)
    loss = reconstruction_mse(t, t)
    assert loss.item() == 0.0


def test_spectral_angle_loss_range():
    a = torch.rand(1, 3, 8, 8)
    b = torch.rand(1, 3, 8, 8)
    loss = spectral_angle_loss(a, b)
    assert loss.item() >= 0.0


def test_band_weighted_mse():
    a = torch.ones(1, 3, 8, 8)
    b = torch.zeros(1, 3, 8, 8)
    weights = torch.tensor([1.0, 0.5, 0.0])
    loss = band_weighted_mse(a, b, weights)
    assert loss.item() > 0.0


def test_spectral_smoothness_loss():
    recon = torch.rand(1, 8, 16, 16)
    loss = spectral_smoothness_loss(recon)
    assert loss.item() >= 0.0


def test_task_aligned_hsi_loss():
    a = torch.rand(1, 4, 16, 16)
    b = torch.rand(1, 4, 16, 16)
    loss = task_aligned_hsi_loss(a, b, {"mse": 1.0, "sam": 0.1})
    assert loss.item() > 0.0


def test_measurement_consistency_loss():
    recon = torch.rand(1, 3, 8, 8)
    measurement = torch.rand(1, 3, 8, 8)
    psf = torch.rand(3, 3, 3)
    loss = measurement_consistency_loss(recon, measurement, psf)
    assert loss.item() >= 0.0
    # Check differentiability
    recon.requires_grad_(True)
    loss2 = measurement_consistency_loss(recon, measurement, psf)
    loss2.backward()
    assert recon.grad is not None


# ── Regularizers ────────────────────────────────────────────────────

def test_optical_param_l2():
    params = [torch.tensor([1.0, 2.0, 3.0])]
    loss = optical_param_l2(params, weight=1.0)
    assert abs(loss.item() - 14.0) < 1e-4


def test_optical_param_l2_empty():
    loss = optical_param_l2([], weight=1.0)
    assert loss.item() == 0.0


def test_rollback_penalty_below_threshold():
    loss = rollback_penalty(rollback_count=2, max_rollbacks=3)
    assert loss.item() == 0.0


def test_rollback_penalty_above_threshold():
    loss = rollback_penalty(rollback_count=5, max_rollbacks=3, penalty_weight=0.1)
    assert loss.item() == pytest.approx(0.2)


def test_psf_energy_preservation_is_differentiable():
    psf = torch.rand(1, 16, 16, requires_grad=True)
    loss = psf_energy_preservation(psf)
    loss.backward()
    assert psf.grad is not None


# ── Objective profiles ──────────────────────────────────────────────

def test_list_objective_profiles():
    profiles = list_objective_profiles()
    assert len(profiles) == 3
    ids = {p.profile_id for p in profiles}
    assert "stable_lens_hsi_codesign" in ids


def test_get_objective_profile():
    p = get_objective_profile("stable_lens_hsi_codesign")
    assert p is not None
    assert "reconstruction_mse" in p.losses
    assert p.claim_implications == "stable_native_lens_hsi_codesign"


def test_get_objective_profile_unknown():
    assert get_objective_profile("nonexistent") is None


def test_preset_profiles_have_valid_backends():
    p = PRESET_PROFILES["stable_lens_hsi_codesign"]
    assert "deeplens_geolens_geometric" in p.compatible_backends


def test_objective_profile_model():
    p = ObjectiveProfile(
        profile_id="test",
        losses=["reconstruction_mse"],
        weights={"reconstruction_mse": 1.0},
        compatible_backends=["mock_deeplens"],
        claim_implications="mock_simulation",
        description="Test profile",
    )
    assert p.profile_id == "test"
    assert p.claim_implications == "mock_simulation"
