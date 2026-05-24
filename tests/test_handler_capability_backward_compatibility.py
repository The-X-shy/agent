"""Test Phase 42 backward compatibility with Phase 40/41 behavior."""

from optiresearch.skills.handler_capability_registry import (
    get_handler_capability_registry,
)


def test_objective_redesign_ceiling_unchanged():
    registry = get_handler_capability_registry()
    cap = registry.get("objective_redesign_simpler_metric")
    assert cap is not None
    assert cap.max_claim_ceiling == "lightweight_scientific_execution"
    assert cap.actual_evidence_level == "lightweight_scientific_execution"
    assert cap.enabled is True
    assert "local" in cap.supported_execution_modes


def test_param_reduction_ceiling_unchanged():
    registry = get_handler_capability_registry()
    cap = registry.get("param_reduction_sweep")
    assert cap is not None
    assert cap.max_claim_ceiling == "lightweight_scientific_execution"
    assert cap.enabled is True


def test_report_negative_result_ceiling_unchanged():
    registry = get_handler_capability_registry()
    cap = registry.get("report_negative_result_doc")
    assert cap is not None
    assert cap.max_claim_ceiling == "report_only"
    assert cap.enabled is True


def test_backend_switch_waveoptics_ceiling_unchanged():
    registry = get_handler_capability_registry()
    cap = registry.get("backend_switch_waveoptics_coherent")
    assert cap is not None
    assert cap.max_claim_ceiling == "structured_unsupported"
    assert "local" not in cap.supported_execution_modes


def test_real_data_request_ceiling_unchanged():
    registry = get_handler_capability_registry()
    cap = registry.get("real_data_request")
    assert cap is not None
    assert cap.max_claim_ceiling == "requires_user_data"
    assert cap.real_data_required is True


def test_all_five_enabled_handlers_present():
    registry = get_handler_capability_registry()
    enabled = registry.list_enabled()
    enabled_ids = {c.handler_id for c in enabled}
    assert "objective_redesign_simpler_metric" in enabled_ids
    assert "param_reduction_sweep" in enabled_ids
    assert "backend_switch_waveoptics_coherent" in enabled_ids
    assert "report_negative_result_doc" in enabled_ids
    assert "real_data_request" in enabled_ids


def test_find_by_design_id_still_works():
    registry = get_handler_capability_registry()
    cap = registry.find_by_design_id("objective_redesign_simpler_metric_mse_only")
    assert cap is not None
    assert cap.handler_id == "objective_redesign_simpler_metric"
