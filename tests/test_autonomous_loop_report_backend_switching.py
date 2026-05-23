"""Test report includes backend switching sections."""

import pytest
from pathlib import Path
from optiresearch.schemas.autonomous_loop import (
    AutonomousLoopResult, AutonomousLoopIteration,
)
from optiresearch.reports.autonomous_loop_report import export_autonomous_loop_report


def test_report_includes_backend_progression(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "workspace").mkdir()

    iterations = [
        AutonomousLoopIteration(
            iteration_id=1,
            strategy_recommendation={"recommended_action": "enable_rollback"},
            execution_result={
                "status": "succeeded",
                "backend_id": "phase_to_fft_proxy",
                "evidence_level": "native_full_reconstruction_proxy",
                "result_payload": {"reconstruction_loss_after": 0.05},
            },
            claim_gate_decision={
                "decision": "supported",
                "max_allowed_claim": "native_full_reconstruction_proxy",
            },
        ),
        AutonomousLoopIteration(
            iteration_id=2,
            strategy_recommendation={"recommended_action": "switch_backend_after_claim_ceiling"},
            execution_result={
                "status": "succeeded",
                "backend_id": "deeplens_geolens_geometric",
                "evidence_level": "native_lens_simulation",
                "result_payload": {"reconstruction_loss_after": 0.03},
            },
            claim_gate_decision={
                "decision": "supported",
                "max_allowed_claim": "native_lens_simulation",
            },
        ),
    ]

    result = AutonomousLoopResult(
        loop_id="test_backend_switch_report",
        status="stopped",
        objective="test backend switching report",
        iterations=iterations,
        trajectory_report_path="claim_ceiling_reached",
    )

    output_dir = tmp_path / "workspace" / "autonomous_loops_v2" / "test_backend_switch_report"
    output_dir.mkdir(parents=True)
    export_autonomous_loop_report(result, output_dir)

    report_path = output_dir / "autonomous_research_loop_report.md"
    content = report_path.read_text()
    assert "Backend Progression" in content
    assert "phase_to_fft_proxy" in content
    assert "deeplens_geolens_geometric" in content


def test_report_includes_evidence_level_progression(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "workspace").mkdir()

    iterations = [
        AutonomousLoopIteration(
            iteration_id=1,
            strategy_recommendation={"recommended_action": "enable_rollback"},
            execution_result={
                "status": "succeeded",
                "backend_id": "phase_to_fft_proxy",
                "evidence_level": "native_full_reconstruction_proxy",
            },
            claim_gate_decision={
                "decision": "supported",
                "max_allowed_claim": "native_full_reconstruction_proxy",
            },
        ),
    ]

    result = AutonomousLoopResult(
        loop_id="test_evidence_report",
        status="completed",
        objective="test evidence report",
        iterations=iterations,
    )

    output_dir = tmp_path / "workspace" / "autonomous_loops_v2" / "test_evidence_report"
    output_dir.mkdir(parents=True)
    export_autonomous_loop_report(result, output_dir)

    report_path = output_dir / "autonomous_research_loop_report.md"
    content = report_path.read_text()
    assert "Evidence Level Progression" in content
