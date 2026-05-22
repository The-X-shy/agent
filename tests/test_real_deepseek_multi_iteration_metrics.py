"""Real DeepSeek multi-iteration local execution test.

Requires:
    OPTIRESEARCH_ENABLE_REAL_LLM_TESTS=1
    DEEPSEEK_API_KEY=<key>
"""

import os
import json
import pytest
from optiresearch.schemas.autonomous_loop import AutonomousLoopSpec
from optiresearch.runtime.autonomous_research_loop import run_autonomous_research_loop


@pytest.mark.skipif(
    os.getenv("OPTIRESEARCH_ENABLE_REAL_LLM_TESTS") != "1"
    or not os.getenv("DEEPSEEK_API_KEY"),
    reason="Real LLM test requires explicit opt-in and DEEPSEEK_API_KEY.",
)
def test_real_deepseek_multi_iteration_local_loop(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "workspace").mkdir()

    spec = AutonomousLoopSpec(
        objective="execute lightweight stable native lens HSI co-design for three iterations",
        max_iterations=3,
        min_iterations_before_stop=2,
        no_improvement_patience=2,
        execution_mode="local",
        allowed_backends=["phase_to_fft_proxy"],
        allowed_task_types=["stable_lens_hsi_codesign"],
        planner_mode="llm_first_with_rule_fallback",
        llm_provider="deepseek",
        prefer_executable_actions=True,
        continue_on_claim_downgrade=True,
        report=True,
        max_runtime_minutes_per_iter=2,
    )

    result = run_autonomous_research_loop(spec)

    assert result.loop_id, "Loop should have an ID"
    assert len(result.iterations) >= 2, (
        f"Expected >= 2 iterations with real LLM, got {len(result.iterations)}. "
        f"Stop reason: {result.trajectory_report_path}"
    )

    for it in result.iterations:
        assert it.execution_result, f"Iteration {it.iteration_id} has no execution result"
        exec_status = it.execution_result.get("status", "")
        assert exec_status in ("succeeded", "claim_downgraded"), (
            f"Iteration {it.iteration_id} unexpected status: {exec_status}"
        )

    metrics_valid_count = 0
    for it in result.iterations:
        payload = it.execution_result.get("result_payload") or it.metrics_snapshot or {}
        if payload.get("reconstruction_loss_after") is not None:
            metrics_valid_count += 1
    assert metrics_valid_count >= 2, (
        f"Expected >= 2 iterations with valid metrics, got {metrics_valid_count}"
    )

    # API key leakage check
    result_json = result.model_dump(mode="json")
    result_str = json.dumps(result_json)
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if api_key:
        assert api_key not in result_str, "API KEY LEAK DETECTED in loop result"

    loop_dir = tmp_path / "workspace" / "autonomous_loops_v2" / result.loop_id
    if loop_dir.exists():
        for json_file in loop_dir.rglob("*.json"):
            content = json_file.read_text()
            if api_key:
                assert api_key not in content, f"API KEY LEAK in {json_file}"

    # Print summary for manual verification
    print(f"\n=== Real DeepSeek Multi-Iteration Test Results ===")
    print(f"Loop ID: {result.loop_id}")
    print(f"Status: {result.status}")
    print(f"Iterations: {len(result.iterations)}")
    for it in result.iterations:
        action = it.strategy_recommendation.get("recommended_action", "N/A")
        status = it.execution_result.get("status", "N/A")
        payload = it.execution_result.get("result_payload") or {}
        loss = payload.get("reconstruction_loss_after", "N/A")
        improved = payload.get("improvement_detected", "N/A")
        claim_dec = (it.claim_gate_decision or {}).get("decision", "N/A")
        print(f"  Iter {it.iteration_id}: action={action}, status={status}, "
              f"loss={loss}, improved={improved}, claim={claim_dec}")
    print(f"Stop reason: {result.trajectory_report_path}")
