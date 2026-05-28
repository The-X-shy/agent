"""Tests for AgentPlanExecutionLoop component pivot validation."""

from unittest.mock import MagicMock, patch

import pytest


def _design(design_id, task_type="diagnostic_probe", spec_payload=None):
    """Build a minimal ExperimentDesignCandidate-like object."""
    d = MagicMock()
    d.design_id = design_id
    d.task_type = task_type
    d.spec_payload = spec_payload or {}
    d.backend_id = "deeplens_fresnel_component"
    d.handler_id = "deeplens_component_first_probe"
    return d


class TestComponentPivotRouting:
    def test_is_diagnostic_design_detects_component_first(self):
        from optiresearch.runtime.agent_plan_execution_loop import _is_diagnostic_design

        d = _design("component_first_fresnel_probe")
        assert _is_diagnostic_design(d) is True

    def test_is_diagnostic_design_detects_component_level(self):
        from optiresearch.runtime.agent_plan_execution_loop import _is_diagnostic_design

        d = _design("component_level_geolens_design")
        assert _is_diagnostic_design(d) is True

    def test_extract_component_from_design_id(self):
        from optiresearch.runtime.agent_plan_execution_loop import _extract_component_from_design

        d = _design("component_first_fresnel_probe")
        assert _extract_component_from_design(d) == "fresnel"

    def test_extract_component_from_binary2phase_design_id(self):
        from optiresearch.runtime.agent_plan_execution_loop import _extract_component_from_design

        d = _design("component_first_binary2phase_probe")
        assert _extract_component_from_design(d) == "binary2phase"

    def test_extract_component_from_spec_payload(self):
        from optiresearch.runtime.agent_plan_execution_loop import _extract_component_from_design

        d = _design("some_other_design", spec_payload={"component": "diffractive"})
        assert _extract_component_from_design(d) == "diffractive"

    def test_extract_component_defaults_to_fresnel(self):
        from optiresearch.runtime.agent_plan_execution_loop import _extract_component_from_design

        d = _design("unknown_design")
        assert _extract_component_from_design(d) == "fresnel"


class TestComponentPivotExecution:
    @patch("optiresearch.runtime.deeplens_component_first_probe.run_deeplens_component_probe")
    def test_component_first_design_local_execution_dispatches(self, mock_probe):
        from optiresearch.runtime.agent_plan_execution_loop import _execute_diagnostic_design
        from optiresearch.schemas.component_probe import ComponentProbeResult

        mock_result = ComponentProbeResult(
            probe_id="test",
            component="fresnel",
            status="succeeded",
            surface_class="Fresnel",
            backend_id="deeplens_fresnel_component",
            differentiable=True,
            parameters_changed=True,
            trainable_param_count=1,
            params_with_grad=1,
            gradient_norm=0.5,
            loss_before=1.0,
            loss_after=0.8,
            claim_ceiling="native_component_optimization",
            evidence_level="diagnostic_evidence",
        )
        mock_probe.return_value = mock_result

        d = _design("component_first_fresnel_probe")
        result = _execute_diagnostic_design(d)
        assert result["status"] == "completed"
        assert result["evidence_level"] == "diagnostic_evidence"
        assert result["metrics"]["component"] == "fresnel"

    @patch("optiresearch.runtime.deeplens_component_first_probe.run_deeplens_component_probe")
    def test_component_first_execution_failure_is_handled(self, mock_probe):
        from optiresearch.runtime.agent_plan_execution_loop import _execute_diagnostic_design

        mock_probe.side_effect = RuntimeError("Probe failed")
        d = _design("component_first_fresnel_probe")
        result = _execute_diagnostic_design(d)
        assert result["status"] != "completed"
