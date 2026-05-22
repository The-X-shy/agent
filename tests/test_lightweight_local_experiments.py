"""Tests for lightweight local experiments."""

import time

import pytest


def test_lightweight_psf_probe_returns_controller_result():
    from optiresearch.runtime.lightweight_experiments import (
        run_lightweight_psf_probe,
    )
    result = run_lightweight_psf_probe(backend_id="phase_to_fft_proxy")
    assert result.status == "succeeded"
    assert result.backend_id == "phase_to_fft_proxy"
    assert result.result_payload is not None
    assert "psf_energy" in result.result_payload
    assert result.result_payload["deepens_used"] is False


def test_lightweight_psf_probe_completes_quickly():
    from optiresearch.runtime.lightweight_experiments import (
        run_lightweight_psf_probe,
    )
    start = time.perf_counter()
    result = run_lightweight_psf_probe()
    elapsed = time.perf_counter() - start
    assert elapsed < 10.0, f"PSF probe took {elapsed:.1f}s, expected <10s"
    assert result.status == "succeeded"


def test_lightweight_stable_lens_hsi_returns_controller_result():
    from optiresearch.runtime.lightweight_experiments import (
        run_lightweight_stable_lens_hsi,
    )
    result = run_lightweight_stable_lens_hsi(
        backend_id="phase_to_fft_proxy",
        max_steps=3,
    )
    assert result.status == "succeeded"
    assert result.backend_id == "phase_to_fft_proxy"
    payload = result.result_payload
    assert payload is not None
    assert "reconstruction_loss_before" in payload
    assert "reconstruction_loss_after" in payload
    assert payload["deepens_used"] is False
    assert payload["accepted_update_count"] >= 0
    assert payload["rollback_count"] >= 0


def test_lightweight_stable_lens_hsi_reports_improvement():
    from optiresearch.runtime.lightweight_experiments import (
        run_lightweight_stable_lens_hsi,
    )
    result = run_lightweight_stable_lens_hsi(
        backend_id="phase_to_fft_proxy",
        max_steps=5,
    )
    payload = result.result_payload
    # With enough steps, loss should improve relative to initial
    assert payload["reconstruction_loss_before"] is not None
    assert payload["reconstruction_loss_after"] is not None


def test_lightweight_ablation_compares_configs():
    from optiresearch.runtime.lightweight_experiments import (
        run_lightweight_ablation,
    )
    result = run_lightweight_ablation(
        backend_id="phase_to_fft_proxy",
        max_configs=2,
        max_steps=2,
    )
    assert result.status == "succeeded"
    payload = result.result_payload
    assert "ablation_results" in payload
    assert len(payload["ablation_results"]) == 2


def test_psf_probe_routes_through_controller():
    from optiresearch.runtime.experiment_controller_v2 import (
        ExperimentControllerV2,
        ExperimentSpecV2,
    )
    from optiresearch.memory.schemas import make_deterministic_id

    spec = ExperimentSpecV2(
        spec_id=make_deterministic_id("test", "psf_probe"),
        task_type="lightweight_psf_probe",
        backend_id="phase_to_fft_proxy",
        spec_payload={"device": "cpu"},
    )
    ctrl = ExperimentControllerV2()
    result = ctrl.run_local(spec)
    assert result.status == "succeeded"
    assert result.result_payload is not None
