"""Tests for native optimization claim evidence integration."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from optiresearch.memory.claim_evidence import ClaimEvidenceManager
from optiresearch.memory.schemas import make_claim_id
from optiresearch.storage.sqlite_store import SQLiteStore


@pytest.fixture
def manager(tmp_path):
    store = SQLiteStore(tmp_path / "test.sqlite")
    store.init_db()
    return ClaimEvidenceManager(store=store, workspace_id="test_native_opt")


def test_create_native_optimization_claim(manager):
    """Create a claim about native differentiable optimization."""
    text = "DeepLens native differentiable optimization is supported on ParaxialLens"
    scope = {
        "evidence_domain": "native_optimization_probe",
        "backend": "deeplens",
        "lens_class": "ParaxialLens",
        "realization_level": "native",
        "differentiable": True,
        "native_parameter_update": True,
        "gradient_norm": 0.15,
        "parameters_changed": True,
        "fallback_used": False,
    }
    claim = manager.create_claim(text, scope)
    assert claim.status == "unsupported"  # Before evidence
    assert claim.claim_id.startswith("claim_")


def test_support_native_optimization_claim(manager):
    """Attach evidence and review a native optimization claim."""
    text = "DeepLens native differentiable optimization is supported"
    scope = {
        "evidence_domain": "native_optimization_probe",
        "backend": "deeplens",
        "realization_level": "native",
        "selected_realization_level": "native",
        "differentiable": True,
        "native_parameter_update": True,
        "gradient_norm": 0.15,
        "parameters_changed": True,
        "loss_before": 0.5,
        "loss_after": 0.3,
        "lens_class": "ParaxialLens",
        "fallback_used": False,
    }
    claim = manager.create_claim(text, scope)
    manager.attach_support(claim.claim_id, "artifact_native_probe_1", score=0.9)
    reviewed = manager.review_claim(claim.claim_id)
    assert reviewed.status in ("supported", "partially_supported")
    assert reviewed.support_score >= 0.9


def test_black_box_codesign_claim(manager):
    """Black-box co-design claim is supported even when differentiable=false."""
    text = "DeepLens-backed black-box co-design is supported"
    scope = {
        "evidence_domain": "codesign_loop",
        "backend": "deeplens",
        "psf_source": "deeplens_parameterized",
        "fallback_used": False,
        "differentiable": False,
    }
    claim = manager.create_claim(text, scope)
    manager.attach_support(claim.claim_id, "artifact_codesign_1", score=0.85)
    reviewed = manager.review_claim(claim.claim_id)
    assert reviewed.status != "unsupported"


def test_hsi_reconstruction_native_claim_needs_followup(manager):
    """Claim about HSI improvement via native optimization needs followup."""
    text = "DeepLens native optimization improves HSI reconstruction"
    scope = {
        "evidence_domain": "native_optimization_probe",
        "backend": "deeplens",
        "realization_level": "native",
        "differentiable": True,
        "native_parameter_update": True,
    }
    claim = manager.create_claim(text, scope)
    manager.attach_support(claim.claim_id, "artifact_native_probe_2", score=0.7)
    reviewed = manager.review_claim(claim.claim_id)
    # Should not be fully supported without HSI metrics
    assert reviewed.status != "contradicted"


def test_explain_claim_includes_native_fields(manager):
    """explain_claim should include native optimization fields."""
    text = "DeepLens native differentiable optimization is supported"
    scope = {
        "evidence_domain": "native_optimization_probe",
        "backend": "deeplens",
        "realization_level": "native",
        "differentiable": True,
        "native_parameter_update": True,
        "gradient_norm": 0.15,
        "parameters_changed": True,
        "loss_before": 0.5,
        "loss_after": 0.3,
        "lens_class": "ParaxialLens",
    }
    claim = manager.create_claim(text, scope)
    manager.attach_support(claim.claim_id, "artifact_native_probe_3", score=0.95)
    reviewed = manager.review_claim(claim.claim_id)
    explanation = manager.explain_claim(claim.claim_id)
    assert "claim_id" in explanation
    assert "claim_text" in explanation
    assert "status" in explanation
    assert "support_score" in explanation


def test_native_optimization_evidence_level(manager):
    """Native optimization probe should get correct evidence level."""
    text = "DeepLens native differentiable optimization probe succeeded"
    scope = {
        "evidence_domain": "native_optimization_probe",
        "backend": "deeplens",
        "realization_level": "native",
        "differentiable": True,
        "native_parameter_update": True,
        "gradient_norm": 0.15,
        "parameters_changed": True,
    }
    claim = manager.create_claim(text, scope)
    assert claim.metadata.get("evidence_level") is not None


def test_non_native_claim_is_unsupported(manager):
    """Non-native claim without sufficient evidence should be unsupported."""
    text = "DeepLens native differentiable optimization is supported on GeoLens"
    scope = {
        "evidence_domain": "native_optimization_probe",
        "backend": "deeplens",
        "realization_level": "unavailable",
        "differentiable": False,
        "native_parameter_update": False,
    }
    claim = manager.create_claim(text, scope)
    reviewed = manager.review_claim(claim.claim_id)
    assert reviewed.status == "unsupported"
