"""Test remote handler config activation."""

from optiresearch.skills.handler_capability_registry import get_handler_capability_registry


def test_remote_native_geolens_is_enabled():
    registry = get_handler_capability_registry()
    cap = registry.get("remote_native_geolens_validation")
    assert cap is not None
    assert cap.enabled is True
    assert cap.supports_remote is True
    assert cap.remote_required is True
    assert cap.requires_remote_validation is True


def test_remote_native_geolens_has_correct_ceilings():
    registry = get_handler_capability_registry()
    cap = registry.get("remote_native_geolens_validation")
    assert cap.remote_evidence_ceiling == "native_lens_simulation"
    assert cap.local_evidence_ceiling == "needs_followup"


def test_remote_native_geolens_execution_modes():
    registry = get_handler_capability_registry()
    cap = registry.get("remote_native_geolens_validation")
    assert "remote_opt_in" in cap.supported_execution_modes
    assert "local" not in cap.supported_execution_modes


def test_enabled_count_is_at_least_6():
    registry = get_handler_capability_registry()
    enabled = registry.list_enabled()
    assert len(enabled) >= 6  # includes remote_native_geolens_validation + DeepLens handlers
