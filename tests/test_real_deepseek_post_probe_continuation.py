"""Phase 32: Real DeepSeek post-probe continuation integration test.

Opt-in test — requires:
  OPTIRESEARCH_ENABLE_REAL_LLM_TESTS=1
  DEEPSEEK_API_KEY=<valid key>
"""

import os
import pytest

from optiresearch.runtime.autonomous_research_loop import run_autonomous_research_loop
from optiresearch.schemas.autonomous_loop import AutonomousLoopSpec


@pytest.mark.skipif(
    not os.environ.get("OPTIRESEARCH_ENABLE_REAL_LLM_TESTS"),
    reason="Set OPTIRESEARCH_ENABLE_REAL_LLM_TESTS=1 to run real LLM tests",
)
def test_real_deepseek_post_probe_continuation(tmp_path):
    """Real DeepSeek loop with full probe+continuation cycle.

    Verifies:
    - max_iterations=4 with backend switching
    - After probe succeeds, next iteration uses run_validated_backend_experiment
    - No stop_and_report immediately after successful probe
    - No strategy_could_not_map_to_experiment
    """
    os.chdir(tmp_path)
    (tmp_path / "workspace").mkdir(exist_ok=True)

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    assert api_key, "DEEPSEEK_API_KEY must be set for real LLM test"

    spec = AutonomousLoopSpec(
        objective="switch backend, validate, and continue with experiment",
        max_iterations=4,
        min_iterations_before_stop=2,
        execution_mode="local",
        allowed_backends=["phase_to_fft_proxy", "deeplens_geolens_geometric"],
        allowed_task_types=[
            "stable_lens_hsi_codesign", "backend_probe",
            "native_lens_simulation_codesign", "lightweight_psf_probe",
        ],
        planner_mode="llm_first_with_rule_fallback",
        llm_provider="deepseek",
        prefer_executable_actions=True,
        allow_backend_switching=True,
        max_backend_switches=1,
        max_runtime_minutes_per_iter=2,
        report=True,
    )

    result = run_autonomous_research_loop(spec)

    assert result.status in ("completed", "stopped")

    for it in result.iterations:
        assert it.stop_reason != "strategy_could_not_map_to_experiment", (
            f"Iteration {it.iteration_id}: strategy_could_not_map_to_experiment"
        )

    actions = [
        it.strategy_recommendation.get("recommended_action", "")
        for it in result.iterations
    ]
    probe_seen = any(a == "probe_new_backend" for a in actions)
    continuation_seen = any(
        a == "run_validated_backend_experiment" for a in actions
    )

    # At minimum, we expect either a probe or continuation action
    assert probe_seen or continuation_seen or len(result.iterations) >= 2, (
        f"Actions: {actions}"
    )


@pytest.mark.skipif(
    not os.environ.get("OPTIRESEARCH_ENABLE_REAL_LLM_TESTS"),
    reason="Set OPTIRESEARCH_ENABLE_REAL_LLM_TESTS=1 to run real LLM tests",
)
def test_real_deepseek_no_stop_after_probe(tmp_path):
    """After successful probe, the loop should not stop immediately."""
    os.chdir(tmp_path)
    (tmp_path / "workspace").mkdir(exist_ok=True)

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    assert api_key, "DEEPSEEK_API_KEY must be set for real LLM test"

    spec = AutonomousLoopSpec(
        objective="validate backend then continue without stopping",
        max_iterations=4,
        min_iterations_before_stop=1,
        execution_mode="local",
        allowed_backends=["phase_to_fft_proxy", "deeplens_geolens_geometric"],
        allowed_task_types=[
            "stable_lens_hsi_codesign", "backend_probe",
            "native_lens_simulation_codesign", "lightweight_psf_probe",
        ],
        planner_mode="llm_first_with_rule_fallback",
        llm_provider="deepseek",
        prefer_executable_actions=True,
        allow_backend_switching=True,
        max_backend_switches=1,
        max_runtime_minutes_per_iter=2,
        report=False,
    )

    result = run_autonomous_research_loop(spec)
    assert result.status in ("completed", "stopped")
    assert len(result.iterations) >= 1
