"""Phase 32: Strategy-to-Spec validated backend experiment tests."""

from optiresearch.agents.strategy_engine import StrategyRecommendation
from optiresearch.agents.strategy_to_spec import compile_experiment_spec


def test_run_validated_backend_experiment_maps_to_native_lens_simulation():
    rec = StrategyRecommendation(
        recommended_action="run_validated_backend_experiment",
        rationale="test",
    )
    spec = compile_experiment_spec(
        rec, "deeplens_geolens_geometric", prefer_executable=True,
    )
    assert spec is not None
    assert spec.task_type == "native_lens_simulation_codesign"


def test_continuation_for_proxy_backend():
    rec = StrategyRecommendation(
        recommended_action="run_validated_backend_experiment",
        rationale="test",
    )
    spec = compile_experiment_spec(
        rec, "phase_to_fft_proxy", prefer_executable=True,
    )
    assert spec is not None
    assert spec.task_type == "stable_lens_hsi_codesign"


def test_continuation_payload_has_lightweight_mode():
    rec = StrategyRecommendation(
        recommended_action="run_validated_backend_experiment",
        rationale="test",
    )
    spec = compile_experiment_spec(
        rec, "deeplens_geolens_geometric", prefer_executable=True,
    )
    assert spec.spec_payload.get("lightweight_mode") is True
    assert spec.spec_payload.get("max_steps") == 3
    assert spec.spec_payload.get("rollback_on_loss_increase") is True


def test_continuation_for_component_backend():
    rec = StrategyRecommendation(
        recommended_action="run_validated_backend_experiment",
        rationale="test",
    )
    spec = compile_experiment_spec(
        rec, "deeplens_fresnel_component", prefer_executable=True,
    )
    assert spec is not None
    assert spec.task_type == "component_optimization"
