"""Phase 31: Strategy-to-Spec probe_new_backend tests."""

from optiresearch.agents.strategy_engine import StrategyRecommendation
from optiresearch.agents.strategy_to_spec import compile_experiment_spec


def test_probe_new_backend_maps_to_backend_probe():
    rec = StrategyRecommendation(
        recommended_action="probe_new_backend",
        rationale="test probe",
    )
    spec = compile_experiment_spec(rec, "phase_to_fft_proxy", prefer_executable=True)
    assert spec is not None
    assert spec.task_type == "backend_probe"


def test_probe_new_backend_payload_has_lightweight_mode():
    rec = StrategyRecommendation(
        recommended_action="probe_new_backend",
        rationale="test probe",
    )
    spec = compile_experiment_spec(rec, "deeplens_geolens_geometric", prefer_executable=True)
    assert spec is not None
    assert spec.spec_payload.get("lightweight_mode") is True
    assert spec.spec_payload.get("max_steps") == 1
    assert spec.spec_payload.get("device") == "cpu"


def test_probe_new_backend_maps_for_deeplens_backend():
    rec = StrategyRecommendation(
        recommended_action="probe_new_backend",
        rationale="test probe",
    )
    spec = compile_experiment_spec(rec, "deeplens_geolens_geometric", prefer_executable=True)
    assert spec is not None
    assert spec.task_type == "backend_probe"
    assert spec.backend_id == "deeplens_geolens_geometric"
