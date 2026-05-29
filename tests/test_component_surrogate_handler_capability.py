"""Handler capability tests for component surrogate HSI co-design."""

from optiresearch.skills.handler_capability_registry import get_handler_capability_registry


def test_component_surrogate_handler_capability_registered():
    cap = get_handler_capability_registry().get("component_surrogate_hsi_codesign")

    assert cap is not None
    assert cap.enabled is True
    assert "local" in cap.supported_execution_modes
    assert "remote_opt_in" in cap.supported_execution_modes
    assert cap.supports_remote is True
    assert cap.remote_required is False
    assert cap.actual_evidence_level == "component_surrogate_hsi_codesign"
    assert cap.max_claim_ceiling == "component_surrogate_hsi_codesign"
    assert cap.synthetic_only is True
    assert cap.physical_backend is False
    assert cap.native_backend_required is False
    assert "reconstruction_loss" in cap.metrics_supported
    assert "component_surrogate_hsi_codesign" in cap.compatible_design_ids
