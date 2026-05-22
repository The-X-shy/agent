"""Tests for enhanced multi-iteration trajectory report."""

import json
from pathlib import Path


def test_report_includes_llm_proposal_table(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "workspace").mkdir()

    from optiresearch.schemas.autonomous_loop import (
        AutonomousLoopIteration,
        AutonomousLoopResult,
    )
    from optiresearch.reports.autonomous_loop_report import (
        export_autonomous_loop_report,
    )

    result = AutonomousLoopResult(
        loop_id="test_report",
        status="completed",
        objective="test",
        iterations=[
            AutonomousLoopIteration(
                iteration_id=1,
                strategy_recommendation={
                    "recommended_action": "retry_with_smaller_lr",
                    "metadata": {
                        "planner": "llm",
                        "proposal_id": "p1",
                        "hypothesis": "Small LR improves stability",
                    },
                },
                execution_result={"status": "succeeded", "result_payload": {"reconstruction_loss_after": 0.3}},
            ),
        ],
        best_result={"status": "succeeded", "result_payload": {"reconstruction_loss_after": 0.3}},
    )

    path = export_autonomous_loop_report(result, tmp_path)
    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert "LLM Proposals" in content
    assert "retry_with_smaller_lr" in content


def test_report_includes_best_iteration(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "workspace").mkdir()

    from optiresearch.schemas.autonomous_loop import (
        AutonomousLoopIteration,
        AutonomousLoopResult,
    )
    from optiresearch.reports.autonomous_loop_report import (
        export_autonomous_loop_report,
    )

    result = AutonomousLoopResult(
        loop_id="test_best",
        status="completed",
        objective="test",
        iterations=[
            AutonomousLoopIteration(
                iteration_id=1,
                execution_result={"status": "succeeded", "backend_id": "phase_to_fft_proxy"},
            ),
        ],
        best_result={"status": "succeeded", "backend_id": "phase_to_fft_proxy"},
    )

    path = export_autonomous_loop_report(result, tmp_path)
    content = path.read_text(encoding="utf-8")
    assert "Best Iteration" in content
    assert "phase_to_fft_proxy" in content


def test_report_includes_claim_evolution(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "workspace").mkdir()

    from optiresearch.schemas.autonomous_loop import (
        AutonomousLoopIteration,
        AutonomousLoopResult,
    )
    from optiresearch.reports.autonomous_loop_report import (
        export_autonomous_loop_report,
    )

    result = AutonomousLoopResult(
        loop_id="test_claim",
        status="completed",
        objective="test",
        iterations=[
            AutonomousLoopIteration(
                iteration_id=1,
                execution_result={"status": "succeeded"},
                claim_gate_decision={
                    "decision": "supported",
                    "max_allowed_claim": "native_lens_simulation",
                },
            ),
        ],
    )

    path = export_autonomous_loop_report(result, tmp_path)
    content = path.read_text(encoding="utf-8")
    assert "Claim Evolution" in content
    assert "supported" in content
