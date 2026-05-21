"""Tests for Phase 21 full reconstruction claim evidence."""

import pytest
from optiresearch.memory.claim_evidence import ClaimEvidenceManager


@pytest.fixture
def manager():
    return ClaimEvidenceManager(workspace_id="test_phase21_recon")


def test_full_reconstruction_claim_can_be_supported(manager):
    claim = manager.create_claim(
        "DeepLens native differentiable optical-HSI reconstruction co-design is supported",
        scope={
            "evidence_domain": "deeplens_native_optimization",
            "native_optimization_level": "optical_hsi_codesign",
            "surface_probe_succeeded": True,
            "requires_grad_true": True,
            "gradient_norm": 0.001,
            "parameters_changed": True,
            "optimizer_step_executed": True,
            "hsi_loss_after": 0.09,
            "full_reconstruction_loss_used": True,
            "recon_gradient_norm": 2.87,
            "phase_to_fft_proxy_used": True,
        },
    )
    manager.attach_support(claim.claim_id, "artifact_recon_1", 0.85)
    manager.review_claim(claim.claim_id)
    reviewed = manager.get_claim(claim.claim_id)
    assert reviewed.status == "supported"


def test_full_reconstruction_without_recon_gradient_is_needs_followup(manager):
    claim = manager.create_claim(
        "DeepLens full native optical-HSI reconstruction co-design is supported",
        scope={
            "evidence_domain": "deeplens_native_optimization",
            "native_optimization_level": "component",
            "surface_probe_succeeded": True,
            "requires_grad_true": True,
            "gradient_norm": 0.001,
            "parameters_changed": True,
            "optimizer_step_executed": True,
            "hsi_loss_after": 0.09,
            "full_reconstruction_loss_used": False,
            "recon_gradient_norm": 0,
        },
    )
    manager.review_claim(claim.claim_id)
    reviewed = manager.get_claim(claim.claim_id)
    assert reviewed.status == "needs_followup"


def test_wave_optics_claim_is_not_yet_supported(manager):
    claim = manager.create_claim(
        "DeepLens full wave-optics native HSI co-design is supported",
        scope={
            "evidence_domain": "deeplens_native_optimization",
            "native_optimization_level": "optical_hsi_codesign",
        },
    )
    manager.review_claim(claim.claim_id)
    reviewed = manager.get_claim(claim.claim_id)
    assert reviewed.status != "supported"


def test_real_hsi_claim_still_unsupported(manager):
    claim = manager.create_claim(
        "DeepLens native optimization improves real HSI performance",
        scope={
            "evidence_domain": "deeplens_native_optimization",
        },
    )
    manager.review_claim(claim.claim_id)
    reviewed = manager.get_claim(claim.claim_id)
    assert reviewed.status != "supported"


def test_proxy_claim_still_works(manager):
    claim = manager.create_claim(
        "DeepLens native differentiable optical-HSI proxy co-design is supported",
        scope={
            "evidence_domain": "deeplens_native_optimization",
            "native_optimization_level": "component",
            "surface_probe_succeeded": True,
            "requires_grad_true": True,
            "gradient_norm": 0.001,
            "parameters_changed": True,
            "optimizer_step_executed": True,
            "hsi_loss_after": 0.09,
        },
    )
    manager.attach_support(claim.claim_id, "artifact_proxy_1", 0.80)
    manager.review_claim(claim.claim_id)
    reviewed = manager.get_claim(claim.claim_id)
    assert reviewed.status == "supported"
