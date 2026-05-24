"""Test candidate plan evaluator remote awareness."""

from optiresearch.agents.candidate_plan_evaluator import CandidatePlanEvaluator
from optiresearch.agents.experiment_design_generator import ExperimentDesignCandidate


def _design(design_id, handler_id="", estimated_runtime_sec=600):
    return ExperimentDesignCandidate(
        design_id=design_id,
        objective="test",
        backend_id="deeplens_geolens_geometric",
        task_type="stable_lens_hsi_codesign",
        handler_id=handler_id,
        estimated_runtime_sec=estimated_runtime_sec,
        risk_level="low",
    )


def test_remote_required_handler_skipped_in_local_mode():
    evaluator = CandidatePlanEvaluator()
    d = _design("remote_native_geolens_validation", handler_id="remote_native_geolens_validation")
    scores = evaluator.evaluate([d])
    selection = evaluator.select_executable_designs(scores, [d], mode="local", limit=1)
    assert len(selection.selected_designs) == 0
    assert selection.stop_reason == "no_executable_design"


def test_remote_required_handler_selected_in_remote_mode():
    evaluator = CandidatePlanEvaluator()
    d = _design("remote_native_geolens_validation", handler_id="remote_native_geolens_validation")
    scores = evaluator.evaluate([d])
    selection = evaluator.select_executable_designs(scores, [d], mode="remote_opt_in", limit=1, allow_remote=True)
    assert len(selection.selected_designs) == 1
    assert selection.selected_designs[0].design_id == "remote_native_geolens_validation"


def test_remote_handler_not_selected_when_allow_remote_false():
    evaluator = CandidatePlanEvaluator()
    d = _design("remote_native_geolens_validation", handler_id="remote_native_geolens_validation")
    scores = evaluator.evaluate([d])
    selection = evaluator.select_executable_designs(scores, [d], mode="remote_opt_in", limit=1, allow_remote=False)
    assert len(selection.selected_designs) == 0


def test_local_handler_still_works_in_remote_mode():
    evaluator = CandidatePlanEvaluator()
    d = _design("objective_redesign_simpler_metric_mse_only", handler_id="objective_redesign_simpler_metric")
    scores = evaluator.evaluate([d])
    selection = evaluator.select_executable_designs(scores, [d], mode="remote_opt_in", limit=1, allow_remote=True)
    assert len(selection.selected_designs) == 1
