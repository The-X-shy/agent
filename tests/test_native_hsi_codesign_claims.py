"""Tests for Phase 20 native HSI co-design claim evidence."""

import pytest
from optiresearch.memory.claim_evidence import ClaimEvidenceManager


@pytest.fixture
def manager():
    return ClaimEvidenceManager(workspace_id="test_hsi_codesign")


def test_component_claim_still_supported(manager):
    claim = manager.create_claim(
        "DeepLens native differentiable component optimization is supported",
        scope={
            "evidence_domain": "deeplens_native_optimization",
            "native_optimization_level": "component",
            "surface_probe_succeeded": True,
            "requires_grad_true": True,
            "gradient_norm": 0.001,
            "parameters_changed": True,
            "optimizer_step_executed": True,
        },
    )
    manager.attach_support(claim.claim_id, "artifact_surface_1", 0.85)
    manager.review_claim(claim.claim_id)
    reviewed = manager.get_claim(claim.claim_id)
    assert reviewed.status == "supported"


def test_proxy_hsi_codesign_claim_can_be_supported(manager):
    claim = manager.create_claim(
        "DeepLens native differentiable optical-HSI proxy co-design is supported",
        scope={
            "evidence_domain": "deeplens_native_optimization",
            "native_optimization_level": "component",
            "evidence_level": "native_hsi_proxy",
            "surface_probe_succeeded": True,
            "requires_grad_true": True,
            "gradient_norm": 0.001,
            "parameters_changed": True,
            "optimizer_step_executed": True,
            "hsi_loss_after": 0.09,
        },
    )
    manager.attach_support(claim.claim_id, "artifact_hsi_proxy_1", 0.80)
    manager.review_claim(claim.claim_id)
    reviewed = manager.get_claim(claim.claim_id)
    assert reviewed.status == "supported"


def test_proxy_hsi_codesign_without_component_chain_needs_followup(manager):
    claim = manager.create_claim(
        "DeepLens native differentiable optical-HSI proxy co-design is supported",
        scope={
            "evidence_domain": "deeplens_native_optimization",
            "native_optimization_level": "component",
            "surface_probe_succeeded": False,
            "gradient_norm": 0,
        },
    )
    manager.review_claim(claim.claim_id)
    reviewed = manager.get_claim(claim.claim_id)
    assert reviewed.status == "needs_followup"


def test_full_reconstruction_claim_without_hsi_chain_is_needs_followup(manager):
    claim = manager.create_claim(
        "DeepLens full native optical-HSI reconstruction co-design is supported",
        scope={
            "evidence_domain": "deeplens_native_optimization",
            "native_optimization_level": "component",
        },
    )
    manager.review_claim(claim.claim_id)
    reviewed = manager.get_claim(claim.claim_id)
    assert reviewed.status == "needs_followup"


def test_real_hsi_claim_is_unsupported(manager):
    claim = manager.create_claim(
        "DeepLens native optimization improves real HSI performance",
        scope={
            "evidence_domain": "deeplens_native_optimization",
            "native_optimization_level": "optical_hsi_codesign",
        },
    )
    manager.review_claim(claim.claim_id)
    reviewed = manager.get_claim(claim.claim_id)
    assert reviewed.status != "supported"
