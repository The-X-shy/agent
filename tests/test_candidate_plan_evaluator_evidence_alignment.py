"""Test candidate plan evaluator evidence alignment penalties."""

from optiresearch.agents.candidate_plan_evaluator import CandidatePlanEvaluator
from optiresearch.agents.experiment_design_generator import ExperimentDesignCandidate


def _design(design_id, expected_evidence, actual_evidence, alignment, backend_id="deeplens_geolens_geometric"):
    return ExperimentDesignCandidate(
        design_id=design_id,
        objective="test",
        backend_id=backend_id,
        task_type="stable_lens_hsi_codesign",
        expected_evidence_level=expected_evidence,
        actual_handler_evidence_level=actual_evidence,
        evidence_alignment_status=alignment,
        estimated_runtime_sec=60,
        risk_level="low",
    )


def test_aligned_design_gets_normal_score():
    evaluator = CandidatePlanEvaluator()
    d = _design("aligned_design", "lightweight_scientific_execution", "lightweight_scientific_execution", "aligned")
    scores = evaluator.evaluate([d])
    assert len(scores) == 1
    assert scores[0].total_score > 0.3


def test_downgraded_design_gets_penalty():
    evaluator = CandidatePlanEvaluator()
    d_aligned = _design("aligned", "lightweight_scientific_execution", "lightweight_scientific_execution", "aligned")
    d_downgraded = _design("downgraded", "lightweight_scientific_execution", "lightweight_scientific_execution", "downgraded_to_handler_capability")
    scores = evaluator.evaluate([d_aligned, d_downgraded])
    # The downgraded design should score lower than the aligned design
    aligned_score = [s for s in scores if s.design_id == "aligned"][0].total_score
    downgraded_score = [s for s in scores if s.design_id == "downgraded"][0].total_score
    assert downgraded_score < aligned_score


def test_needs_followup_has_low_feasibility():
    evaluator = CandidatePlanEvaluator()
    d = _design("followup", "structured_unsupported", "needs_followup", "downgraded_to_handler_capability")
    scores = evaluator.evaluate([d])
    breakdown = scores[0].score_breakdown
    assert breakdown["execution_feasibility"] <= 0.1


def test_requires_user_data_has_zero_feasibility():
    evaluator = CandidatePlanEvaluator()
    d = _design("data_req", "requires_user_data", "requires_user_data", "aligned")
    scores = evaluator.evaluate([d])
    breakdown = scores[0].score_breakdown
    assert breakdown["execution_feasibility"] <= 0.01
    assert breakdown["metric_gain_likelihood"] <= 0.01


def test_lightweight_scientific_has_high_metric_likelihood():
    evaluator = CandidatePlanEvaluator()
    d = _design("lightweight", "lightweight_scientific_execution", "lightweight_scientific_execution", "aligned")
    scores = evaluator.evaluate([d])
    breakdown = scores[0].score_breakdown
    assert breakdown["metric_gain_likelihood"] > 0.1
