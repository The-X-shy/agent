"""Test two scientific handlers can execute and compare."""

from optiresearch.runtime.agent_plan_execution_loop import (
    _execute_design,
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


def test_both_scientific_handlers_produce_metrics():
    mse_result = _execute_design(
        _candidate(
            "objective_redesign_simpler_metric_mse_only",
            spec_payload={
                "loss_weights": {"mse": 1.0, "spectral_angle": 0.0, "measurement_consistency": 0.0},
                "max_steps": 3,
            },
        )
    )
    pr_result = _execute_design(
        _candidate(
            "param_reduction_sweep",
            spec_payload={"param_subset": ["curvature"]},
        )
    )

    assert mse_result["status"] == "completed"
    assert mse_result["evidence_level"] == "lightweight_scientific_execution"
    assert pr_result["status"] == "completed"
    assert pr_result["evidence_level"] == "lightweight_scientific_execution"

    # Both produce real metrics
    for result in (mse_result, pr_result):
        assert "reconstruction_loss_before" in result["metrics"]
        assert "reconstruction_loss_after" in result["metrics"]
        assert "mse_after" in result["metrics"]
        assert "improvement_detected" in result["metrics"]


def test_handlers_produce_different_metrics():
    """Each handler has its own metric keys."""
    mse_result = _execute_design(
        _candidate(
            "objective_redesign_simpler_metric_mse_only",
            spec_payload={
                "loss_weights": {"mse": 1.0, "spectral_angle": 0.0, "measurement_consistency": 0.0},
                "max_steps": 3,
            },
        )
    )
    pr_result = _execute_design(
        _candidate(
            "param_reduction_sweep",
            spec_payload={"param_subset": ["curvature"]},
        )
    )

    # Param reduction has configs_tested and best_k
    assert "configs_tested" in pr_result["metrics"]
    assert "best_k" in pr_result["metrics"]

    # MSE-only has mse_only_objective
    assert mse_result["metrics"].get("mse_only_objective") is True
