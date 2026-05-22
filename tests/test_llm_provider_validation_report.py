"""Tests for LLM provider validation report."""

import json
from pathlib import Path

import pytest

from optiresearch.reports.llm_provider_validation_report import (
    export_llm_provider_validation_report,
)


def test_report_generates_with_no_data(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "workspace").mkdir()

    output_dir = tmp_path / "output"
    path = export_llm_provider_validation_report(
        planner_run_id="nonexistent",
        loop_id="nonexistent",
        output_dir=output_dir,
    )
    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert "No provider check data available" in content
    assert "No planner trace data available" in content
    assert "No loop result data available" in content


def test_report_populated_sections(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "reports").mkdir(parents=True)
    (ws / "planner_traces" / "planner_001").mkdir(parents=True)
    (ws / "autonomous_loops_v2" / "loop_001").mkdir(parents=True)

    # Write provider check
    (ws / "reports" / "llm_provider_check.json").write_text(json.dumps({
        "status": "available",
        "provider": "deepseek",
        "model": "deepseek-v4-pro",
        "base_url": "https://api.deepseek.com",
        "error_code": None,
        "latency_ms": 123.4,
    }))

    # Write planner trace
    (ws / "planner_traces" / "planner_001" / "validation_report.json").write_text(
        json.dumps([
            {"proposal_id": "p1", "valid": True, "errors": []},
            {"proposal_id": "p2", "valid": False, "errors": ["backend not found"]},
        ])
    )
    (ws / "planner_traces" / "planner_001" / "selected_proposal.json").write_text(
        json.dumps({
            "proposal_id": "p1",
            "proposed_claim": "Improves optimization.",
            "safe_wording": "Improves optimization.",
        })
    )

    # Write loop result
    (ws / "autonomous_loops_v2" / "loop_001" / "loop_result.json").write_text(
        json.dumps({
            "loop_id": "loop_001",
            "status": "completed",
            "objective": "test",
            "total_iterations": 2,
            "iterations": [
                {"id": "iter_1", "action": "retry_with_smaller_lr", "exec_status": "completed", "next_action": "continue", "stop_reason": None},
                {"id": "iter_2", "action": "stop_and_report", "exec_status": "completed", "next_action": None, "stop_reason": "claim_ceiling_reached"},
            ],
            "final_supported_claims": ["Improves optimization."],
            "final_unsupported_claims": [],
            "trajectory_report_path": "workspace/autonomous_loops_v2/loop_001/trajectory.md",
        })
    )

    output_dir = tmp_path / "output"
    path = export_llm_provider_validation_report(
        planner_run_id="planner_001",
        loop_id="loop_001",
        output_dir=output_dir,
    )
    assert path.exists()
    content = path.read_text(encoding="utf-8")

    # Verify sections are populated
    assert "## 1. Provider Environment Status" in content
    assert "deepseek-v4-pro" in content
    assert "**p1:** PASS" in content or "p1: PASS" in content
    assert "**p2:** FAIL" in content or "p2: FAIL" in content
    assert "retry_with_smaller_lr" in content
