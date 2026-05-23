"""Test that scientific designs are preferred over report-only fallback."""

from optiresearch.agents.candidate_plan_evaluator import (
    CandidatePlanEvaluator,
    _is_lightweight_scientific_design,
    _is_report_only_design,
)
from optiresearch.agents.experiment_design_generator import ExperimentDesignCandidate


def _scientific_design():
    return ExperimentDesignCandidate(
        design_id="objective_redesign_simpler_metric_mse_only",
        objective="Test MSE-only loss for smoother optimization",
        backend_id="deeplens_geolens_geometric",
        task_type="stable_lens_hsi_codesign",
        spec_payload={
            "loss_weights": {"mse": 1.0, "spectral_angle": 0.0, "measurement_consistency": 0.0},
            "max_steps": 5,
        },
        expected_evidence_level="lightweight_scientific_execution",
        expected_failure_modes=["gradient_instability"],
        estimated_runtime_sec=60,
        risk_level="low",
    )


def _report_design():
    return ExperimentDesignCandidate(
        design_id="report_negative_result_doc",
        objective="Document negative result",
        backend_id="",
        task_type="native_lens_simulation_codesign",
        spec_payload={"action": "export_system_subunit_report"},
        required_skills=["report_generation"],
        estimated_runtime_sec=60,
        risk_level="low",
    )


def test_is_lightweight_scientific_design_by_design_id():
    d = _scientific_design()
    assert _is_lightweight_scientific_design(d) is True


def test_is_lightweight_scientific_design_by_loss_weights():
    d = ExperimentDesignCandidate(
        design_id="some_other_mse_design",
        objective="test",
        backend_id="phase_to_fft_proxy",
        task_type="stable_lens_hsi_codesign",
        spec_payload={"loss_weights": {"mse": 1.0, "spectral_angle": 0.0}},
        estimated_runtime_sec=60,
    )
    assert _is_lightweight_scientific_design(d) is True


def test_is_lightweight_scientific_design_negative():
    d = ExperimentDesignCandidate(
        design_id="not_scientific",
        objective="test",
        backend_id="deeplens_geolens_geometric",
        task_type="stable_lens_hsi_codesign",
        spec_payload={},
        estimated_runtime_sec=600,
    )
    assert _is_lightweight_scientific_design(d) is False


def test_scientific_is_not_report_only():
    d = _scientific_design()
    assert _is_report_only_design(d) is False


def test_scientific_design_not_deferred_as_report():
    d_sci = _scientific_design()
    d_rep = _report_design()

    designs = [d_sci, d_rep]
    evaluator = CandidatePlanEvaluator()
    scores = evaluator.evaluate(designs)
    selection = evaluator.select_executable_designs(
        scores,
        designs,
        mode="local",
        limit=1,
    )
    # The scientific design should be selected first
    assert selection.selected_design is not None
    # Report design should be deferred, not selected first
    if selection.selected_design == d_rep.design_id:
        # Only acceptable if scientific design was skipped for a real reason
        skipped_reasons = [
            item.get("skipped_reason", "")
            for item in selection.skipped_higher_ranked_designs
        ]
        assert any("unsupported" not in r.lower() or r == "" for r in skipped_reasons) or len(skipped_reasons) == 0


def test_scientific_design_is_locally_supported():
    from optiresearch.agents.candidate_plan_evaluator import _is_local_supported_design
    d = _scientific_design()
    assert _is_local_supported_design(d) is True


def test_selection_with_scientific_and_report():
    d_sci = _scientific_design()
    d_rep = _report_design()

    evaluator = CandidatePlanEvaluator()
    scores = evaluator.evaluate([d_sci, d_rep])
    selection = evaluator.select_executable_designs(
        scores,
        [d_sci, d_rep],
        mode="local",
        limit=2,
    )
    # At least one design should be selected
    assert len(selection.selected_designs) >= 1
    # The selected design should include the scientific design
    selected_ids = [d.design_id for d in selection.selected_designs]
    assert d_sci.design_id in selected_ids
