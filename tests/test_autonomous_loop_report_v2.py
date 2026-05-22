"""Phase 25 autonomous loop report tests."""

import tempfile
from pathlib import Path

from optiresearch.schemas.autonomous_loop import (
    AutonomousLoopIteration,
    AutonomousLoopResult,
)
from optiresearch.reports.autonomous_loop_report import export_autonomous_loop_report


def _make_result() -> AutonomousLoopResult:
    it = AutonomousLoopIteration(
        iteration_id=1,
        strategy_recommendation={
            "recommended_action": "retry_with_smaller_lr",
            "rationale": "Large gradients",
            "risk_level": "low",
        },
        execution_result={
            "status": "succeeded",
            "result_payload": {"reconstruction_loss_after": 0.5},
        },
        claim_gate_decision={
            "decision": "supported",
            "max_allowed_claim": "native_lens_simulation",
        },
        next_action="stop",
        stop_reason="max_iterations_reached",
    )
    return AutonomousLoopResult(
        loop_id="test_loop_report",
        status="completed",
        objective="Test objective",
        iterations=[it],
        final_supported_claims=["Native lens simulation is supported"],
        final_unsupported_claims=[],
        trajectory_report_path="max_iterations_reached",
    )


def test_report_exports_markdown(tmp_path):
    result = _make_result()
    path = export_autonomous_loop_report(result, tmp_path)
    assert path.exists()
    content = path.read_text()
    assert "Autonomous Research Loop Report" in content
    assert "test_loop_report" in content
    assert "Test objective" in content


def test_report_contains_iteration_summary(tmp_path):
    result = _make_result()
    path = export_autonomous_loop_report(result, tmp_path)
    content = path.read_text()
    assert "Iteration Summary" in content
    assert "retry_with_smaller_lr" in content


def test_report_contains_strategy_decisions(tmp_path):
    result = _make_result()
    path = export_autonomous_loop_report(result, tmp_path)
    content = path.read_text()
    assert "Strategy Decisions" in content
    assert "Large gradients" in content


def test_report_contains_claim_gate_decisions(tmp_path):
    result = _make_result()
    path = export_autonomous_loop_report(result, tmp_path)
    content = path.read_text()
    assert "Claim Gate Decisions" in content


def test_report_contains_final_claim_status(tmp_path):
    result = _make_result()
    path = export_autonomous_loop_report(result, tmp_path)
    content = path.read_text()
    assert "Final Claim Status" in content
    assert "Native lens simulation" in content


def test_report_contains_stop_reason(tmp_path):
    result = _make_result()
    path = export_autonomous_loop_report(result, tmp_path)
    content = path.read_text()
    assert "Stop Reason" in content
    assert "max_iterations_reached" in content


def test_report_with_empty_iterations(tmp_path):
    result = AutonomousLoopResult(
        loop_id="empty_loop",
        status="stopped",
        objective="No iterations",
        trajectory_report_path="no_iterations",
    )
    path = export_autonomous_loop_report(result, tmp_path)
    content = path.read_text()
    assert "No iterations" in content
    assert "empty_loop" in content
