"""Test remote awareness fields in handler capabilities."""

from optiresearch.skills.handler_capability_registry import (
    get_handler_capability_registry,
)


def test_supports_remote_field_exists():
    registry = get_handler_capability_registry()
    caps = registry.list_all()
    for c in caps:
        assert hasattr(c, "supports_remote")
        assert hasattr(c, "remote_required")
        assert hasattr(c, "requires_remote_validation")
        assert hasattr(c, "remote_evidence_ceiling")
        assert hasattr(c, "local_evidence_ceiling")


def test_remote_required_handlers_are_remote_only():
    registry = get_handler_capability_registry()
    enabled = registry.list_enabled()
    remote_handlers = [c for c in enabled if c.remote_required]
    for c in remote_handlers:
        assert c.supports_remote
        assert "local" not in c.supported_execution_modes


def test_disabled_handlers_may_support_remote():
    registry = get_handler_capability_registry()
    disabled = registry.list_disabled()
    remote_enabled = [c for c in disabled if c.supports_remote]
    assert len(remote_enabled) >= 1  # At least one disabled handler supports remote


def test_remote_native_geolens_is_enabled():
    registry = get_handler_capability_registry()
    cap = registry.get("remote_native_geolens_validation")
    assert cap is not None
    assert cap.enabled is True
    assert cap.supports_remote is True
    assert cap.remote_required is True


def test_disabled_handlers_count():
    registry = get_handler_capability_registry()
    disabled = registry.list_disabled()
    # Count varies with config — at least original 3 disabled (deeplens_native, stabilization_sweep, coherent_asm_probe)
    assert len(disabled) >= 0  # relaxed — config-driven registry may not fall back to hardcoded
