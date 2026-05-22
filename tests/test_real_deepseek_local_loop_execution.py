"""Real DeepSeek local loop execution test.

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
def test_real_deepseek_local_loop_execution(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "workspace").mkdir()

    from optiresearch.schemas.autonomous_loop import AutonomousLoopSpec
    from optiresearch.runtime.autonomous_research_loop import (
        run_autonomous_research_loop,
    )

    spec = AutonomousLoopSpec(
        objective="investigate executable lightweight experiments without overclaiming",
        max_iterations=2,
        execution_mode="local",
        planner_mode="llm_first_with_rule_fallback",
        llm_provider="deepseek",
        prefer_executable_actions=True,
        allowed_backends=["phase_to_fft_proxy"],
        allowed_task_types=[
            "stable_lens_hsi_codesign",
            "lightweight_psf_probe",
        ],
        max_llm_proposals=3,
        memory_update=False,
        report=False,
    )

    result = run_autonomous_research_loop(spec)

    # Must have completed or stopped
    assert result.status in ("completed", "stopped"), (
        f"Unexpected status: {result.status}"
    )

    # Must have at least one iteration
    assert len(result.iterations) >= 1

    # At least one iteration should have an execution result
    executed = [
        it for it in result.iterations
        if it.execution_result and it.execution_result.get("status") is not None
    ]
    assert len(executed) >= 1, (
        "Expected at least one iteration with execution result"
    )

    # Should have supported/unsupported claims lists
    assert result.final_supported_claims is not None
    assert result.final_unsupported_claims is not None

    # Verify no API key in any output file
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if api_key:
        loop_dir = (
            tmp_path / "workspace" / "autonomous_loops_v2" / result.loop_id
        )
        if loop_dir.exists():
            for f in loop_dir.rglob("*.json"):
                content = f.read_text(encoding="utf-8")
                assert api_key not in content, (
                    f"API key found in {f.name}"
                )
