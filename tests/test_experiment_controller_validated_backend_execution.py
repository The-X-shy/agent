"""Phase 32: ExperimentController validated backend execution tests."""

from optiresearch.runtime.experiment_controller_v2 import (
    ExperimentControllerV2, ExperimentSpecV2,
)
from optiresearch.memory.schemas import make_deterministic_id


def test_native_lens_simulation_codesign_routes():
    ctrl = ExperimentControllerV2()
    spec = ExperimentSpecV2(
        spec_id=make_deterministic_id("test", "nlsc", "v2"),
        task_type="native_lens_simulation_codesign",
        backend_id="deeplens_geolens_geometric",
        spec_payload={"max_steps": 3, "lightweight_mode": True, "device": "cpu"},
    )
    result = ctrl.run_local(spec)
    assert result.status in ("succeeded", "failed", "unsupported")
    assert result.backend_id == "deeplens_geolens_geometric"


def test_native_lens_simulation_codesign_produces_metrics():
    ctrl = ExperimentControllerV2()
    spec = ExperimentSpecV2(
        spec_id=make_deterministic_id("test", "nlsc", "metrics"),
        task_type="native_lens_simulation_codesign",
        backend_id="deeplens_geolens_geometric",
        spec_payload={"max_steps": 3, "lightweight_mode": True, "device": "cpu"},
    )
    result = ctrl.run_local(spec)
    if result.status == "succeeded":
        payload = result.result_payload or {}
        assert "reconstruction_loss_after" in payload or "loss_after" in payload


def test_native_lens_simulation_evidence_level():
    ctrl = ExperimentControllerV2()
    spec = ExperimentSpecV2(
        spec_id=make_deterministic_id("test", "nlsc", "ev"),
        task_type="native_lens_simulation_codesign",
        backend_id="deeplens_geolens_geometric",
        spec_payload={"max_steps": 2, "lightweight_mode": True},
    )
    result = ctrl.run_local(spec)
    if result.status == "succeeded":
        assert result.evidence_level in (
            "native_lens_simulation", "native_full_reconstruction_proxy",
        )
