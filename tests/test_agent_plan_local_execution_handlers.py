from optiresearch.agents.experiment_design_generator import ExperimentDesignCandidate
from optiresearch.runtime.agent_plan_execution_loop import _execute_design


def _candidate(
    design_id: str,
    task_type: str,
    backend_id: str,
    spec_payload: dict | None = None,
    required_skills: list[str] | None = None,
):
    return ExperimentDesignCandidate(
        design_id=design_id,
        objective=design_id,
        backend_id=backend_id,
        task_type=task_type,
        spec_payload=spec_payload or {},
        required_skills=required_skills or [],
    )


def test_report_negative_result_doc_executes_as_report_only():
    result = _execute_design(
        _candidate(
            "report_negative_result_doc",
            task_type="native_lens_simulation_codesign",
            backend_id="",
            spec_payload={"action": "export_system_subunit_report"},
            required_skills=["report_generation"],
        )
    )

    assert result["status"] == "completed"
    assert result["design_id"] == "report_negative_result_doc"
    assert result["evidence_level"] == "report_only"
    assert result["task_type"] == "report_generation"
    assert result["artifacts"]
    assert result["errors"] == []


def test_param_reduction_sweep_returns_structured_unsupported():
    result = _execute_design(
        _candidate(
            "param_reduction_sweep",
            task_type="stable_lens_hsi_codesign",
            backend_id="deeplens_geolens_geometric",
            spec_payload={"param_subset": ["curvature", "thickness"]},
        )
    )

    assert result["status"] == "unsupported"
    assert result["evidence_level"] == "structured_unsupported"
    assert result["errors"][0]["type"] == "HANDLER_MISSING_PARAM_REDUCTION_SWEEP"


def test_backend_switch_waveoptics_probe_returns_needs_followup():
    result = _execute_design(
        _candidate(
            "backend_switch_waveoptics_coherent",
            task_type="native_waveoptics_codesign",
            backend_id="deeplens_coherent_asm",
            spec_payload={"candidate": "GeoLensCooke"},
        )
    )

    assert result["status"] == "needs_followup"
    assert result["evidence_level"] == "structured_unsupported"
    assert result["metrics"]["requires_grad"] is False
    assert result["errors"][0]["type"] == "COHERENT_ASM_REQUIRES_GRAD_FALSE"
