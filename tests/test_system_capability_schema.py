"""Tests for SystemCapabilityEntry and SystemCapabilityRegistry schemas."""
from __future__ import annotations

import pytest

from optiresearch.schemas.system_capability import SystemCapabilityEntry, SystemCapabilityRegistry


def test_system_capability_entry_minimal():
    entry = SystemCapabilityEntry(
        capability_id="test_handler",
        capability_type="handler",
        name="Test Handler",
    )
    assert entry.capability_id == "test_handler"
    assert entry.capability_type == "handler"
    assert entry.enabled is True
    assert entry.evidence_level == "unsupported"
    assert entry.maturity_level == "experimental"


def test_system_capability_entry_full():
    entry = SystemCapabilityEntry(
        capability_id="deeplens_native_geolens_hsi_codesign",
        capability_type="handler",
        name="DeepLens Native GeoLens HSI",
        enabled=True,
        maturity_level="validated_local",
        supported_execution_modes=["local", "remote_opt_in"],
        evidence_level="native_lens_simulation",
        max_claim_ceiling="native_lens_simulation",
        synthetic_only=True,
        native_backend_required=True,
        physical_backend=False,
        supports_remote=True,
        requires_deeplens=True,
        known_limitations=["geometric only"],
        blocked_claims=["real_hsi_performance"],
        safe_wording="synthetic geometric optimization",
        owner_module="optiresearch.runtime.native_geolens_hsi",
    )
    assert entry.supports_remote is True
    assert entry.native_backend_required is True
    assert entry.blocked_claims == ["real_hsi_performance"]


def test_system_capability_entry_rejects_extra_fields():
    with pytest.raises(ValueError):
        SystemCapabilityEntry(
            capability_id="test",
            capability_type="handler",
            unknown_field="should_fail",
        )


def test_system_capability_registry_empty():
    reg = SystemCapabilityRegistry()
    assert reg.registry_version == "0.1"
    assert reg.entries == []


def test_system_capability_registry_create():
    entries = [
        SystemCapabilityEntry(capability_id="h1", capability_type="handler"),
        SystemCapabilityEntry(capability_id="b1", capability_type="backend"),
    ]
    reg = SystemCapabilityRegistry.create(
        entries=entries,
        source_files=["test.yaml"],
        validation_summary={"total_entries": 2},
    )
    assert len(reg.entries) == 2
    assert reg.source_files == ["test.yaml"]
    assert reg.validation_summary["total_entries"] == 2
    assert reg.generated_at != ""
