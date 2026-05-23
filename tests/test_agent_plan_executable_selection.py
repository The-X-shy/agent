from optiresearch.agents.candidate_plan_evaluator import (
    CandidatePlanEvaluator,
    PlanScore,
)
from optiresearch.agents.experiment_design_generator import ExperimentDesignCandidate


def _design(
    design_id: str,
    task_type: str = "stable_lens_hsi_codesign",
    backend_id: str = "deeplens_geolens_geometric",
    estimated_runtime_sec: int = 60,
    required_skills: list[str] | None = None,
):
    return ExperimentDesignCandidate(
        design_id=design_id,
        objective=design_id,
        backend_id=backend_id,
        task_type=task_type,
        estimated_runtime_sec=estimated_runtime_sec,
        required_skills=required_skills or [],
    )


def test_local_selection_skips_needs_user_data_top_ranked_design():
    designs = [
        _design("real_data_request_req", estimated_runtime_sec=0),
        _design("alt_param_diffractive_sweep"),
    ]
    scores = [
        PlanScore("real_data_request_req", 0.9, recommendation="needs_user_data"),
        PlanScore("alt_param_diffractive_sweep", 0.7, recommendation="dry_run_first"),
    ]

    selection = CandidatePlanEvaluator().select_executable_designs(
        scores, designs, mode="local", limit=1
    )

    assert selection.selected_design == "alt_param_diffractive_sweep"
    assert selection.selected_design_rank == 2
    assert selection.selected_designs[0].design_id == "alt_param_diffractive_sweep"
    assert selection.skipped_higher_ranked_designs == [
        {
            "design_id": "real_data_request_req",
            "rank": 1,
            "recommendation": "needs_user_data",
            "skipped_reason": "needs_user_data",
        }
    ]
    assert "highest-ranked local executable" in selection.executable_selection_reason


def test_dry_run_selection_keeps_top_design_for_display():
    designs = [
        _design("real_data_request_req", estimated_runtime_sec=0),
        _design("alt_param_diffractive_sweep"),
    ]
    scores = [
        PlanScore("real_data_request_req", 0.9, recommendation="needs_user_data"),
        PlanScore("alt_param_diffractive_sweep", 0.7, recommendation="dry_run_first"),
    ]

    selection = CandidatePlanEvaluator().select_executable_designs(
        scores, designs, mode="dry_run", limit=1
    )

    assert selection.selected_design == "real_data_request_req"
    assert selection.selected_design_rank == 1
    assert selection.skipped_higher_ranked_designs == []


def test_local_selection_stops_when_no_design_is_executable():
    designs = [
        _design("real_data_request_req", estimated_runtime_sec=0),
        _design("remote_waveoptics_job", estimated_runtime_sec=3600),
    ]
    scores = [
        PlanScore("real_data_request_req", 0.9, recommendation="needs_user_data"),
        PlanScore("remote_waveoptics_job", 0.8, recommendation="needs_remote"),
    ]

    selection = CandidatePlanEvaluator().select_executable_designs(
        scores, designs, mode="local", limit=1
    )

    assert selection.selected_design is None
    assert selection.stop_reason == "no_executable_design"
    assert [s["skipped_reason"] for s in selection.skipped_higher_ranked_designs] == [
        "needs_user_data",
        "needs_remote",
    ]
