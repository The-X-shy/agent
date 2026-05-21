"""Tests for Phase 19B native optimization claim levels."""

from __future__ import annotations

import pytest

from optiresearch.memory.claim_evidence import ClaimEvidenceManager
from optiresearch.storage.sqlite_store import SQLiteStore


@pytest.fixture
def manager(tmp_path):
    store = SQLiteStore(tmp_path / "claims.sqlite")
    store.init_db()
    return ClaimEvidenceManager(store=store, workspace_id="phase19b")


def test_component_level_native_optimization_claim_can_be_supported(manager):
    claim = manager.create_claim(
        "DeepLens native differentiable component optimization is supported",
        {
            "evidence_domain": "deeplens_native_optimization",
            "native_optimization_level": "component",
            "backend": "deeplens",
            "surface_class": "Fresnel",
            "surface_probe_succeeded": True,
            "requires_grad_true": True,
            "gradient_norm": 0.2,
            "parameters_changed": True,
            "optimizer_step_executed": True,
        },
    )
    manager.attach_support(claim.claim_id, "surface_probe_artifact", score=0.9)

    reviewed = manager.review_claim(claim.claim_id)

    assert reviewed.status == "supported"
    assert reviewed.metadata["evidence_level"] == "deeplens_native_component_optimization"


def test_surface_probe_does_not_support_full_optical_hsi_codesign(manager):
    claim = manager.create_claim(
        "DeepLens native optical-HSI co-design is supported",
        {
            "evidence_domain": "deeplens_native_optimization",
            "native_optimization_level": "component",
            "backend": "deeplens",
            "surface_class": "Fresnel",
            "surface_probe_succeeded": True,
            "requires_grad_true": True,
            "gradient_norm": 0.2,
            "parameters_changed": True,
            "optimizer_step_executed": True,
        },
    )
    manager.attach_support(claim.claim_id, "surface_probe_artifact", score=0.95)

    reviewed = manager.review_claim(claim.claim_id)

    assert reviewed.status == "needs_followup"
    assert "native_optical_hsi_codesign_requires_hsi_loss" in reviewed.warnings


def test_lens_level_claim_requires_lens_file_psf_backward(manager):
    claim = manager.create_claim(
        "DeepLens native differentiable lens optimization is supported",
        {
            "evidence_domain": "deeplens_native_optimization",
            "native_optimization_level": "component",
            "backend": "deeplens",
            "surface_probe_succeeded": True,
            "requires_grad_true": True,
            "gradient_norm": 0.2,
            "parameters_changed": True,
            "optimizer_step_executed": True,
        },
    )
    manager.attach_support(claim.claim_id, "surface_probe_artifact", score=0.9)

    reviewed = manager.review_claim(claim.claim_id)

    assert reviewed.status == "needs_followup"
    assert "native_lens_optimization_requires_lensfile_psf_backward" in reviewed.warnings
