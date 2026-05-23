"""Test experiment controller backend switching support."""

import pytest
from optiresearch.runtime.experiment_controller_v2 import (
    ExperimentControllerV2,
    ExperimentSpecV2,
)


def test_geolens_psf_probe_runs_locally():
    ctrl = ExperimentControllerV2()
    spec = ExperimentSpecV2(
        spec_id="test_geolens_psf_probe",
        task_type="psf_probe",
        backend_id="deeplens_geolens_geometric",
        spec_payload={"lightweight_mode": True, "device": "cpu"},
    )
    result = ctrl.run_local(spec)
    assert result.status in ("succeeded", "skipped", "unsupported")


def test_phase_to_fft_proxy_to_geolens_switch_spec():
    ctrl = ExperimentControllerV2()
    spec = ExperimentSpecV2(
        spec_id="test_switch_to_geolens",
        task_type="psf_probe",
        backend_id="deeplens_geolens_geometric",
        spec_payload={"lightweight_mode": True, "device": "cpu", "max_steps": 3},
    )
    result = ctrl.run_local(spec)
    assert result is not None
    assert result.backend_id == "deeplens_geolens_geometric"


def test_component_optimization_routes_to_lightweight():
    ctrl = ExperimentControllerV2()
    spec = ExperimentSpecV2(
        spec_id="test_component_opt",
        task_type="component_optimization",
        backend_id="deeplens_fresnel_component",
        spec_payload={"device": "cpu", "max_steps": 2},
    )
    result = ctrl.run_local(spec)
    assert result.status in ("succeeded", "skipped", "unsupported")
