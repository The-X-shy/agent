"""Real DeepSeek planner smoke test.

Requires explicit opt-in:
  OPTIRESEARCH_ENABLE_REAL_LLM_TESTS=1
  DEEPSEEK_API_KEY=<key>
"""

import json
import os
from pathlib import Path

import pytest


@pytest.mark.skipif(
    os.getenv("OPTIRESEARCH_ENABLE_REAL_LLM_TESTS") != "1"
    or not os.getenv("DEEPSEEK_API_KEY"),
    reason="Real LLM test requires explicit opt-in and DEEPSEEK_API_KEY.",
)
def test_real_deepseek_planner_smoke(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "workspace").mkdir()

    from optiresearch.agents.llm_planner import LLMPlanner

    planner = LLMPlanner()
    result = planner.plan(
        objective="investigate differentiable wave-optics alternatives without overclaiming",
        provider_name="deepseek",
        allowed_backends=[
            "phase_to_fft_proxy",
            "deeplens_geolens_geometric",
            "local_synthetic_hsi",
        ],
        execution_mode="dry_run",
    )

    # Must be either succeeded or fallback_used
    assert result.status in ("succeeded", "fallback_used"), f"Unexpected status: {result.status}"

    if result.status == "succeeded":
        # Proposals count
        assert 1 <= len(result.proposals) <= 3, f"Expected 1-3 proposals, got {len(result.proposals)}"

        # Each proposal must pass schema validation (have required fields)
        for prop in result.proposals:
            assert prop.proposal_id, "Proposal missing proposal_id"
            assert prop.hypothesis, "Proposal missing hypothesis"
            assert prop.backend_id, "Proposal missing backend_id"
            # No shell commands in claims
            for field in [prop.proposed_claim, prop.rationale, prop.hypothesis]:
                assert "`" not in field, f"Shell backtick found in proposal field"

        # Selected proposal should have safe_wording if claim gate applied
        if result.selected_proposal:
            assert result.selected_proposal.safe_wording is not None or result.selected_proposal.proposed_claim, (
                "Selected proposal has neither safe_wording nor proposed_claim"
            )

    # Planner trace must exist
    assert result.planner_trace_path, "No planner trace path returned"
    trace_dir = Path(result.planner_trace_path) if not os.path.isabs(result.planner_trace_path) else Path(result.planner_trace_path)
    if not trace_dir.exists():
        # Try workspace-relative
        trace_dir = tmp_path / "workspace" / "planner_traces" / result.planner_run_id

    # Verify no API key in any trace file
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if trace_dir.exists() and api_key:
        for f in trace_dir.iterdir():
            if f.suffix == ".json":
                content = f.read_text(encoding="utf-8")
                assert api_key not in content, f"API key found in trace file: {f.name}"
