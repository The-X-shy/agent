"""Agent planning tests for component surrogate HSI strategy generation."""

from optiresearch.agents.candidate_plan_evaluator import CandidatePlanEvaluator
from optiresearch.agents.evidence_strategy_reasoner import EvidenceStrategyReasoner
from optiresearch.agents.experiment_design_generator import ExperimentDesignGenerator


def _diagnosis():
    return {
        "status": "diagnosed",
        "diagnosis_id": "phase63",
        "failure_modes": [
            "full_geolens_direct_update_blocked",
            "no_standard_trainable_parameters",
            "autograd_graph_disconnected",
        ],
        "likely_causes": [],
        "recommended_recoveries": [
            "component_probe_succeeded:fresnel",
            "component_probe_succeeded:binary2phase",
        ],
        "severity": "critical",
    }


def test_component_surrogate_strategies_generated_after_blocked_geolens():
    strategies = EvidenceStrategyReasoner().reason_from_diagnosis(_diagnosis(), "recover")
    ids = [s.strategy_id for s in strategies]

    assert "component_surrogate_hsi_codesign_fresnel" in ids
    assert "component_surrogate_hsi_codesign_binary2phase" in ids
    assert all(
        s.blocked_route == "full_geolens_direct_update"
        for s in strategies
        if s.strategy_id.startswith("component_surrogate_hsi_codesign")
    )


def test_component_surrogate_designs_generated_and_aligned():
    strategies = EvidenceStrategyReasoner().reason_from_diagnosis(_diagnosis(), "recover")
    designs = ExperimentDesignGenerator().generate_designs(strategies)
    by_id = {d.design_id: d for d in designs}

    assert "component_surrogate_fresnel_hsi_codesign_design" in by_id
    assert "component_surrogate_binary2phase_hsi_codesign_design" in by_id
    design = by_id["component_surrogate_fresnel_hsi_codesign_design"]
    assert design.handler_id == "component_surrogate_hsi_codesign"
    assert design.expected_evidence_level == "component_surrogate_hsi_codesign"
    assert design.claim_ceiling == "component_surrogate_hsi_codesign"


def test_component_probe_success_boosts_surrogate_hsi_design():
    strategies = EvidenceStrategyReasoner().reason_from_diagnosis(_diagnosis(), "recover")
    designs = ExperimentDesignGenerator().generate_designs(strategies)
    evaluator = CandidatePlanEvaluator()
    evaluator.set_diagnosis_context(_diagnosis())
    scores = evaluator.evaluate(designs)
    score_map = {s.design_id: s for s in scores}

    assert score_map["component_surrogate_fresnel_hsi_codesign_design"].diagnosis_score_bonus > 0


def test_successful_geolens_audit_boosts_native_geolens_design():
    diagnosis = {
        "status": "diagnosed",
        "diagnosis_id": "geolens-audit-success",
        "failure_modes": [],
        "likely_causes": [],
        "recommended_recoveries": [],
        "severity": "medium",
        "trainable_param_count": 14,
        "params_with_grad": 14,
        "psf_requires_grad": True,
        "loss_requires_grad": True,
        "graph_connected": True,
    }
    strategies = EvidenceStrategyReasoner().reason_from_diagnosis(diagnosis, "recover")
    designs = ExperimentDesignGenerator().generate_designs(strategies)
    evaluator = CandidatePlanEvaluator()
    evaluator.set_diagnosis_context(diagnosis)
    scores = evaluator.evaluate(designs)
    score_map = {s.design_id: s for s in scores}

    assert "full_geolens_geometric_direct_update_design" in score_map
    assert score_map["full_geolens_geometric_direct_update_design"].diagnosis_score_bonus > 0
