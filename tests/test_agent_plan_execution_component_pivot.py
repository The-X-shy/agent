"""Test AgentPlanExecutionLoop component pivot behavior."""

import json
from pathlib import Path
from unittest.mock import patch

from optiresearch.agents.experiment_design_generator import ExperimentDesignCandidate
from optiresearch.runtime.agent_plan_execution_loop import (
    _is_diagnostic_design,
    _execute_diagnostic_design,
)


class TestComponentPivotRouting:
    def test_component_first_design_is_diagnostic(self):
        d = ExperimentDesignCandidate(
            design_id="component_level_geolens_probe_design",
            objective="test", backend_id="deeplens_binary2phase_component",
            task_type="backend_probe", required_skills=[],
        )
        assert _is_diagnostic_design(d)

    def test_component_first_design_local_execution_returns_unsupported(self):
        """On macOS without DeepLens component backend, component probe returns unsupported."""
        d = ExperimentDesignCandidate(
            design_id="component_first_fresnel_design",
            objective="test", backend_id="deeplens_fresnel_component",
            task_type="backend_probe", required_skills=["deeplens_component_first_probe"],
        )
        result = _execute_diagnostic_design(d)
        assert result["status"] == "unsupported"
        assert "COMPONENT_BACKEND_UNAVAILABLE" in str(result.get("errors", []))

    def test_diagnostic_design_has_diagnostic_evidence_ceiling(self):
        d = ExperimentDesignCandidate(
            design_id="autograd_graph_audit_design",
            objective="test", backend_id="deeplens_geolens_geometric",
            task_type="autograd_audit", required_skills=[],
        )
        result = _execute_diagnostic_design(d)
        assert result["evidence_level"] == "diagnostic_evidence"
        assert "does not confirm optical design improvement" in str(result.get("caveats", []))

    def test_verify_trainable_ceiling_is_diagnostic(self):
        d = ExperimentDesignCandidate(
            design_id="verify_trainable_parameters_design",
            objective="test", backend_id="deeplens_geolens_geometric",
            task_type="backend_probe", required_skills=[],
        )
        result = _execute_diagnostic_design(d)
        assert result["evidence_level"] == "diagnostic_evidence"
