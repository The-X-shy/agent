"""Tests for SystemCapabilityRegistry builder."""
from __future__ import annotations

from optiresearch.schemas.system_capability import SystemCapabilityRegistry
from optiresearch.system.capability_registry import build_system_capability_registry


def test_build_returns_system_capability_registry():
    reg = build_system_capability_registry()
    assert isinstance(reg, SystemCapabilityRegistry)
    assert reg.registry_version == "0.1"
    assert len(reg.entries) > 0
    assert reg.generated_at != ""


def test_contains_handler_entries():
    reg = build_system_capability_registry()
    handler_entries = [e for e in reg.entries if e.capability_type == "handler"]
    handler_ids = {e.capability_id for e in handler_entries}
    assert len(handler_entries) > 0
    assert "objective_redesign_simpler_metric" in handler_ids
    assert "deeplens_trainable_parameter_inspection" in handler_ids


def test_contains_backend_entries():
    reg = build_system_capability_registry()
    backend_entries = [e for e in reg.entries if e.capability_type == "backend"]
    backend_ids = {e.capability_id for e in backend_entries}
    assert len(backend_entries) > 0
    assert "deeplens_geolens_geometric" in backend_ids
    assert "mock_deeplens" in backend_ids


def test_contains_skill_entries():
    reg = build_system_capability_registry()
    skill_entries = [e for e in reg.entries if e.capability_type == "skill"]
    assert len(skill_entries) > 0
    skill_ids = {e.capability_id for e in skill_entries}
    assert "deeplens_native_geolens_hsi_codesign" in skill_ids


def test_contains_design_entries():
    reg = build_system_capability_registry()
    design_entries = [e for e in reg.entries if e.capability_type == "design"]
    assert len(design_entries) > 0
    design_ids = {e.capability_id for e in design_entries}
    assert "geolens_curriculum_probe" in design_ids


def test_contains_claim_policy_entries():
    reg = build_system_capability_registry()
    policy_entries = [e for e in reg.entries if e.capability_type == "claim_policy"]
    assert len(policy_entries) > 0


def test_validation_summary_has_expected_keys():
    reg = build_system_capability_registry()
    vs = reg.validation_summary
    assert "total_entries" in vs
    assert "by_type" in vs
    assert "handler_ids" in vs
    assert "missing_evidence_level" in vs
    assert "missing_claim_ceiling" in vs


def test_all_entries_have_capability_types():
    reg = build_system_capability_registry()
    valid_types = {"handler", "skill", "design", "backend", "dataset", "remote_worker",
                   "artifact", "report", "benchmark", "claim_policy"}
    for entry in reg.entries:
        assert entry.capability_type in valid_types, f"{entry.capability_id} has invalid type {entry.capability_type}"
