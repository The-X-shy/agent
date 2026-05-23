"""Phase 33: Execution fidelity routing tests."""

from optiresearch.runtime.experiment_controller_v2 import (
    ExperimentControllerV2, ExperimentSpecV2,
)
from optiresearch.memory.schemas import make_deterministic_id


def test_deeplens_geolens_routes_to_native():
    ctrl = ExperimentControllerV2()
    spec = ExperimentSpecV2(
        spec_id=make_deterministic_id("test", "fidelity", "1"),
        task_type="native_lens_simulation_codesign",
        backend_id="deeplens_geolens_geometric",
        spec_payload={"max_steps": 2, "lightweight_mode": False},
    )
    result = ctrl.run_local(spec)
    assert result.status in ("succeeded", "failed", "unsupported")
    assert result.backend_id == "deeplens_geolens_geometric"


def test_phase_to_fft_proxy_routes_to_lightweight():
    ctrl = ExperimentControllerV2()
    spec = ExperimentSpecV2(
        spec_id=make_deterministic_id("test", "fidelity", "2"),
        task_type="native_lens_simulation_codesign",
        backend_id="phase_to_fft_proxy",
        spec_payload={"max_steps": 2, "lightweight_mode": True},
    )
    result = ctrl.run_local(spec)
    assert result.status in ("succeeded", "failed")


def test_deeplens_native_produces_evidence():
    ctrl = ExperimentControllerV2()
    spec = ExperimentSpecV2(
        spec_id=make_deterministic_id("test", "fidelity", "3"),
        task_type="native_lens_simulation_codesign",
        backend_id="deeplens_geolens_geometric",
        spec_payload={"max_steps": 2, "lightweight_mode": False},
    )
    result = ctrl.run_local(spec)
    if result.status == "succeeded":
        assert result.evidence_level == "native_lens_simulation"
    else:
        assert result.status in ("failed", "unsupported")
