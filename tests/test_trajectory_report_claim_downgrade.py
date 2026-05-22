"""Test that the autonomous loop report includes claim downgrade and metric trajectory sections."""

import pytest
from pathlib import Path
from optiresearch.schemas.autonomous_loop import (
    AutonomousLoopResult,
    AutonomousLoopIteration,
)
from optiresearch.reports.autonomous_loop_report import (
    export_autonomous_loop_report,
)


def test_report_includes_claim_downgrade_events(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "workspace").mkdir()

    iterations = [
        AutonomousLoopIteration(
            iteration_id=1,
            strategy_recommendation={"recommended_action": "enable_rollback"},
            execution_result={
                "status": "claim_downgraded",
                "downgraded_from": "native_lens_simulation",
                "downgraded_to": "native_full_reconstruction_proxy",
                "safe_claim_wording": "Proxy FFT-based HSI reconstruction",
                "claim_downgraded": True,
                "result_payload": {
                    "reconstruction_loss_after": 0.05,
                    "improvement_detected": True,
                },
            },
            claim_gate_decision={
                "decision": "qualified",
                "max_allowed_claim": "native_full_reconstruction_proxy",
                "violation_type": "proxy_as_waveoptics",
                "safe_wording": "Proxy FFT-based HSI reconstruction",
            },
        ),
    ]

    result = AutonomousLoopResult(
        loop_id="test_loop",
        status="stopped",
        objective="test claim downgrade report",
        iterations=iterations,
    )

    output_dir = tmp_path / "workspace" / "autonomous_loops_v2" / "test_loop"
    output_dir.mkdir(parents=True)
    export_autonomous_loop_report(result, output_dir)

    report_path = output_dir / "autonomous_research_loop_report.md"
    assert report_path.exists()

    content = report_path.read_text()
    assert "claim" in content.lower()
    assert "downgrad" in content.lower()
    assert "metric" in content.lower()


def test_report_includes_metric_trajectory_data(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "workspace").mkdir()

    iterations = [
        AutonomousLoopIteration(
            iteration_id=1,
            strategy_recommendation={"recommended_action": "enable_rollback"},
            execution_result={
                "status": "succeeded",
                "result_payload": {
                    "reconstruction_loss_before": 0.10,
                    "reconstruction_loss_after": 0.05,
                    "mse_after": 0.05,
                    "psnr_after": 13.01,
                    "improvement_detected": True,
                },
            },
            claim_gate_decision={"decision": "supported"},
        ),
    ]

    result = AutonomousLoopResult(
        loop_id="test_loop2",
        status="completed",
        objective="test metric report",
        iterations=iterations,
    )

    output_dir = tmp_path / "workspace" / "autonomous_loops_v2" / "test_loop2"
    output_dir.mkdir(parents=True)
    export_autonomous_loop_report(result, output_dir)

    report_path = output_dir / "autonomous_research_loop_report.md"
    content = report_path.read_text()
    assert "Metric Trajectory Data" in content
    assert "Loss Before" in content
    assert "Improvement" in content


def test_report_includes_stop_diagnostics(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "workspace").mkdir()

    iterations = [
        AutonomousLoopIteration(
            iteration_id=1,
            stop_reason="max_iterations_reached",
            next_action="stop",
            strategy_recommendation={},
            execution_result={"status": "succeeded"},
        ),
    ]

    result = AutonomousLoopResult(
        loop_id="test_loop3",
        status="stopped",
        objective="test stop diagnostics",
        iterations=iterations,
        trajectory_report_path="max_iterations_3_reached",
    )

    output_dir = tmp_path / "workspace" / "autonomous_loops_v2" / "test_loop3"
    output_dir.mkdir(parents=True)
    export_autonomous_loop_report(result, output_dir)

    report_path = output_dir / "autonomous_research_loop_report.md"
    content = report_path.read_text()
    assert "Stop Condition Diagnostics" in content
    assert "max_iterations" in content


def test_report_includes_spec_patch_table(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "workspace").mkdir()

    iterations = [
        AutonomousLoopIteration(
            iteration_id=1,
            strategy_recommendation={"recommended_action": "enable_rollback"},
            experiment_spec={
                "spec_payload": {
                    "optical_lr": 1e-7,
                    "max_steps": 5,
                    "rollback_on_loss_increase": True,
                },
            },
            execution_result={"status": "succeeded"},
        ),
    ]

    result = AutonomousLoopResult(
        loop_id="test_loop4",
        status="completed",
        objective="test spec patch report",
        iterations=iterations,
    )

    output_dir = tmp_path / "workspace" / "autonomous_loops_v2" / "test_loop4"
    output_dir.mkdir(parents=True)
    export_autonomous_loop_report(result, output_dir)

    report_path = output_dir / "autonomous_research_loop_report.md"
    content = report_path.read_text()
    assert "Experiment Spec Patches" in content
    assert "optical_lr" in content
