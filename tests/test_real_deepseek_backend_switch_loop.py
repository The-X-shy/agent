"""Real DeepSeek backend-switching loop test. Opt-in only.

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
def test_real_deepseek_backend_switch_loop(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "workspace").mkdir()

    spec = AutonomousLoopSpec(
        objective="continue after claim ceiling by switching backend",
        max_iterations=3,
        min_iterations_before_stop=2,
        no_improvement_patience=2,
        execution_mode="local",
        allowed_backends=["phase_to_fft_proxy", "deeplens_geolens_geometric"],
        allowed_task_types=["stable_lens_hsi_codesign", "psf_probe"],
        planner_mode="llm_first_with_rule_fallback",
        llm_provider="deepseek",
        prefer_executable_actions=True,
        allow_backend_switching=True,
        max_backend_switches=1,
        continue_on_claim_downgrade=True,
        report=True,
        max_runtime_minutes_per_iter=2,
    )

    result = run_autonomous_research_loop(spec)

    assert result.loop_id
    assert len(result.iterations) >= 2, (
        f"Expected >= 2 iterations, got {len(result.iterations)}. "
        f"Stop: {result.trajectory_report_path}"
    )

    backends_seen = set()
    for it in result.iterations:
        assert it.execution_result, f"Iter {it.iteration_id} no exec result"
        bid = it.execution_result.get("backend_id", "")
        if bid:
            backends_seen.add(bid)

    assert "phase_to_fft_proxy" in backends_seen

    # API key leakage
    result_str = json.dumps(result.model_dump(mode="json"))
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if api_key:
        assert api_key not in result_str, "API KEY LEAK"

    print(f"\n=== Real DeepSeek Backend-Switch Test ===")
    print(f"Loop: {result.loop_id}, Status: {result.status}")
    print(f"Iterations: {len(result.iterations)}, Backends: {sorted(backends_seen)}")
    for it in result.iterations:
        action = it.strategy_recommendation.get("recommended_action", "N/A")
        status = it.execution_result.get("status", "N/A")
        bid = it.execution_result.get("backend_id", "N/A")
        payload = it.execution_result.get("result_payload") or {}
        loss = payload.get("reconstruction_loss_after", "N/A")
        print(f"  Iter {it.iteration_id}: backend={bid}, action={action}, "
              f"status={status}, loss={loss}")
    print(f"Stop: {result.trajectory_report_path}")
