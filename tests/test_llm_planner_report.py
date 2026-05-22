"""Phase 26 LLM planner report tests."""

from optiresearch.reports.llm_planner_report import export_llm_planner_report
from optiresearch.agents.llm_planner import LLMPlanner


def test_report_exports_for_active_trace(tmp_path):
    # First create a planner trace
    planner = LLMPlanner()
    result = planner.plan("report test objective", provider_name="mock")

    path = export_llm_planner_report(result.planner_run_id, tmp_path)
    assert path.exists()
    content = path.read_text()
    assert "LLM Planner Report" in content
    assert result.planner_run_id in content
