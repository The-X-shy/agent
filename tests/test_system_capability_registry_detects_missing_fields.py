"""Tests that the registry builder detects missing fields."""
from __future__ import annotations

from optiresearch.system.capability_registry import build_system_capability_registry
from optiresearch.schemas.system_capability import SystemCapabilityEntry


def test_builder_marks_entries_without_owner_module():
    reg = build_system_capability_registry()
    vs = reg.validation_summary
    assert "missing_owner_module" in vs
    # All entries should have owner_module (we set it for all collector functions)
    assert vs["missing_owner_module"] == 0


def test_enabled_handlers_have_evidence_level():
    reg = build_system_capability_registry()
    handlers = [e for e in reg.entries if e.capability_type == "handler" and e.enabled]
    for h in handlers:
        assert h.evidence_level, f"Handler {h.capability_id} missing evidence_level"


def test_all_entries_have_valid_maturity_level():
    reg = build_system_capability_registry()
    valid_maturities = {"experimental", "validated_local", "validated_remote", "benchmarked", "production_ready"}
    for entry in reg.entries:
        assert entry.maturity_level in valid_maturities, \
            f"{entry.capability_id} has invalid maturity {entry.maturity_level}"


def test_handler_entries_have_proper_capability_id_format():
    reg = build_system_capability_registry()
    handlers = [e for e in reg.entries if e.capability_type == "handler"]
    for h in handlers:
        assert h.capability_id, "Handler entry has empty capability_id"
        assert " " not in h.capability_id, f"Handler {h.capability_id} has spaces in id"


def test_backend_entries_have_known_limitations():
    reg = build_system_capability_registry()
    backends = [e for e in reg.entries if e.capability_type == "backend"]
    for b in backends:
        assert isinstance(b.known_limitations, list), \
            f"Backend {b.capability_id} known_limitations is not a list"
