from optiresearch.runtime import agent_plan_execution_loop as loop
from optiresearch.schemas.agent_plan_execution import AgentPlanExecutionSpec


def test_local_mode_attempts_selected_design_then_report_fallback(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def fake_execute(design):
        if design.design_id == "report_negative_result_doc":
            return {
                "status": "completed",
                "design_id": design.design_id,
                "task_type": "report_generation",
                "backend_id": "",
                "evidence_level": "report_only",
                "metrics": {"report_generated": True},
                "artifacts": ["workspace/reports/negative_result.md"],
                "errors": [],
                "caveats": ["Report-only evidence does not support optical improvement"],
            }
        return {
            "status": "unsupported",
            "design_id": design.design_id,
            "task_type": design.task_type,
            "backend_id": design.backend_id,
            "evidence_level": "structured_unsupported",
            "metrics": {},
            "artifacts": [],
            "errors": [{"type": "DEEPLENS_UNAVAILABLE", "message": "not importable"}],
            "caveats": ["Scientific execution unavailable locally"],
        }

    monkeypatch.setattr(loop, "_execute_design", fake_execute)

    result = loop.run_agent_plan_execution(
        AgentPlanExecutionSpec(
            execution_id="plan_exec_local_test",
            objective="recover from native GeoLens optical update instability",
            mode="local",
            execute_top_k=1,
        )
    )

    assert result.status == "completed"
    assert result.selected_design
    assert result.selected_design_rank is not None
    assert result.attempted_designs[0]["design_id"] == result.selected_design
    assert result.attempted_designs[-1]["design_id"] == "report_negative_result_doc"
    assert result.execution_result["status"] == "completed"
    assert result.execution_result["evidence_level"] == "report_only"
    assert result.fallback_to_report_only is True
    assert result.memory_updated is True
    assert result.state_snapshots_count > 0
    assert result.event_count >= 7
    assert result.claim_gate_decision["decision"] == "supported"


def test_local_mode_stops_without_executable_design(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    class NoExecutableEvaluator(loop.CandidatePlanEvaluator):
        def select_executable_designs(self, scores, designs, mode, limit=1, allow_remote=False):
            from optiresearch.agents.candidate_plan_evaluator import ExecutableDesignSelection

            return ExecutableDesignSelection(
                mode=mode,
                selected_designs=[],
                skipped_higher_ranked_designs=[],
                executable_selection_reason="No executable design found",
                stop_reason="no_executable_design",
            )

    monkeypatch.setattr(loop, "CandidatePlanEvaluator", NoExecutableEvaluator)

    result = loop.run_agent_plan_execution(
        AgentPlanExecutionSpec(
            execution_id="plan_exec_no_exec_test",
            objective="recover from native GeoLens optical update instability",
            mode="local",
            execute_top_k=1,
        )
    )

    assert result.status == "stopped"
    assert result.stop_reason == "no_executable_design"
    assert result.execution_result == {}
