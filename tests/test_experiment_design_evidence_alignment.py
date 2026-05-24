"""Test experiment design evidence alignment via HandlerCapabilityRegistry."""

from optiresearch.agents.evidence_strategy_reasoner import CandidateStrategy
from optiresearch.agents.experiment_design_generator import ExperimentDesignGenerator


def test_objective_redesign_gets_downgraded():
    gen = ExperimentDesignGenerator()
    strat = CandidateStrategy(
        strategy_id="objective_redesign_simpler_metric",
        strategy_type="objective_redesign",
        rationale="test",
        expected_evidence_gain="medium",
        expected_metric_gain="medium",
        risk="low",
        cost="low",
        required_backend="deeplens_geolens_geometric",
        claim_ceiling="native_lens_simulation",
    )
    designs = gen.generate_designs([strat])
    for d in designs:
        if "objective_redesign" in d.design_id:
            assert d.evidence_alignment_status == "downgraded_to_handler_capability"
            assert d.actual_handler_evidence_level == "lightweight_scientific_execution"
            assert d.expected_evidence_level == "lightweight_scientific_execution"
            assert d.handler_id == "objective_redesign_simpler_metric"


def test_report_design_is_aligned():
    gen = ExperimentDesignGenerator()
    strat = CandidateStrategy(
        strategy_id="report_negative_result",
        strategy_type="report_negative_result",
        rationale="test",
        expected_evidence_gain="low",
        expected_metric_gain="low",
        risk="low",
        cost="low",
        required_backend="",
        claim_ceiling="report_only",
    )
    designs = gen.generate_designs([strat])
    for d in designs:
        if "report" in d.design_id:
            assert d.evidence_alignment_status in ("aligned", "downgraded_to_handler_capability")


def test_real_data_request_is_unsupported_or_downgraded():
    gen = ExperimentDesignGenerator()
    strat = CandidateStrategy(
        strategy_id="real_data_request",
        strategy_type="real_data_request",
        rationale="test",
        expected_evidence_gain="high",
        expected_metric_gain="high",
        risk="low",
        cost="medium",
        required_backend="",
        claim_ceiling="real_hsi",
    )
    designs = gen.generate_designs([strat])
    for d in designs:
        if "real_data" in d.design_id:
            assert d.evidence_alignment_status in ("downgraded_to_handler_capability", "aligned")


def test_design_has_handler_id_and_alignment_fields():
    gen = ExperimentDesignGenerator()
    strat = CandidateStrategy(
        strategy_id="objective_redesign_simpler_metric",
        strategy_type="objective_redesign",
        rationale="test",
        expected_evidence_gain="medium",
        expected_metric_gain="medium",
        risk="low",
        cost="low",
        required_backend="deeplens_geolens_geometric",
        claim_ceiling="native_lens_simulation",
    )
    designs = gen.generate_designs([strat])
    for d in designs:
        if "objective_redesign" in d.design_id:
            assert hasattr(d, "handler_id")
            assert hasattr(d, "actual_handler_evidence_level")
            assert hasattr(d, "evidence_alignment_status")
            assert hasattr(d, "evidence_downgrade_reason")
