"""Test Handler Capability Registry registration and queries."""

from optiresearch.skills.handler_capability_registry import (
    HandlerCapability,
    HandlerCapabilityRegistry,
    get_handler_capability_registry,
)


def test_registry_has_five_builtins():
    registry = HandlerCapabilityRegistry()
    caps = registry.list_all()
    assert len(caps) == 5


def test_find_by_design_id_objective_redesign():
    registry = HandlerCapabilityRegistry()
    cap = registry.find_by_design_id("objective_redesign_simpler_metric_mse_only")
    assert cap is not None
    assert cap.handler_id == "objective_redesign_simpler_metric"
    assert cap.actual_evidence_level == "lightweight_scientific_execution"
    assert cap.synthetic_only is True
    assert cap.physical_backend is False


def test_find_by_design_id_param_reduction():
    registry = HandlerCapabilityRegistry()
    cap = registry.find_by_design_id("param_reduction_sweep")
    assert cap is not None
    assert cap.handler_id == "param_reduction_sweep"
    assert cap.actual_evidence_level == "lightweight_scientific_execution"
    assert "local" in cap.supported_execution_modes


def test_find_by_design_id_backend_switch():
    registry = HandlerCapabilityRegistry()
    cap = registry.find_by_design_id("backend_switch_waveoptics_coherent")
    assert cap is not None
    assert cap.actual_evidence_level == "structured_unsupported"
    assert cap.metrics_supported == []
    assert "local" not in cap.supported_execution_modes


def test_find_by_design_id_report():
    registry = HandlerCapabilityRegistry()
    cap = registry.find_by_design_id("report_negative_result_doc")
    assert cap is not None
    assert cap.actual_evidence_level == "report_only"
    assert "report_generated" in cap.metrics_supported


def test_find_by_design_id_real_data():
    registry = HandlerCapabilityRegistry()
    cap = registry.find_by_design_id("real_data_request_req")
    assert cap is not None
    assert cap.actual_evidence_level == "requires_user_data"
    assert cap.real_data_required is True


def test_get_actual_evidence_level():
    registry = HandlerCapabilityRegistry()
    assert registry.get_actual_evidence_level("objective_redesign_simpler_metric_mse_only") == "lightweight_scientific_execution"
    assert registry.get_actual_evidence_level("nonexistent") is None


def test_is_locally_executable():
    registry = HandlerCapabilityRegistry()
    assert registry.is_locally_executable("objective_redesign_simpler_metric_mse_only") is True
    assert registry.is_locally_executable("param_reduction_sweep") is True
    assert registry.is_locally_executable("report_negative_result_doc") is True
    assert registry.is_locally_executable("backend_switch_waveoptics_coherent") is False


def test_inspect():
    registry = HandlerCapabilityRegistry()
    info = registry.inspect("objective_redesign_simpler_metric")
    assert info is not None
    assert info["handler_id"] == "objective_redesign_simpler_metric"
    assert "mse_before" in info["metrics_supported"]


def test_singleton():
    r1 = get_handler_capability_registry()
    r2 = get_handler_capability_registry()
    assert r1 is r2
