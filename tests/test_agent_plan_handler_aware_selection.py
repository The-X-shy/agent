"""Test agent plan handler-aware selection."""

from optiresearch.runtime.agent_plan_execution_loop import (
    _execute_design,
    _execute_param_reduction_sweep,
)
from optiresearch.agents.experiment_design_generator import ExperimentDesignCandidate


def _candidate(design_id, spec_payload=None):
    return ExperimentDesignCandidate(
        design_id=design_id,
        objective=design_id,
        backend_id="deeplens_geolens_geometric",
        task_type="stable_lens_hsi_codesign",
        spec_payload=spec_payload or {},
    )


def test_param_reduction_design_executes_via_handler():
    result = _execute_design(
        _candidate(
            "param_reduction_sweep",
            spec_payload={"param_subset": ["curvature", "thickness"]},
        )
    )
    assert result["status"] == "completed"
    assert result["evidence_level"] == "lightweight_scientific_execution"
    assert "configs_tested" in result["metrics"]
    assert "best_k" in result["metrics"]


def test_param_reduction_produces_handler_id():
    result = _execute_param_reduction_sweep(
        _candidate("param_reduction_sweep")
    )
    assert result["handler_id"] == "param_reduction_sweep"


def test_param_reduction_has_caveats():
    result = _execute_design(
        _candidate("param_reduction_sweep")
    )
    assert result["caveats"]
    assert any("synthetic" in c.lower() for c in result["caveats"])
