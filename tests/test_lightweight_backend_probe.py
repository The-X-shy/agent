"""Phase 31: Lightweight backend probe tests."""

import time

from optiresearch.runtime.lightweight_experiments import run_lightweight_backend_probe


def test_phase_to_fft_proxy_probe_succeeds():
    result = run_lightweight_backend_probe(backend_id="phase_to_fft_proxy")
    assert result.status == "succeeded"
    assert result.result_payload is not None
    assert result.result_payload.get("backend_available") is True
    assert result.result_payload.get("probe_status") == "succeeded"
    assert "probe_time_seconds" in result.result_payload


def test_mock_deeplens_probe_succeeds():
    result = run_lightweight_backend_probe(backend_id="mock_deeplens")
    assert result.status == "succeeded"
    assert result.result_payload.get("backend_available") is True


def test_deeplens_geolens_probe_never_crashes():
    result = run_lightweight_backend_probe(backend_id="deeplens_geolens_geometric")
    assert result.status in ("succeeded", "failed")
    assert result.result_payload is not None
    assert "probe_time_seconds" in result.result_payload
    assert result.result_payload.get("backend_id") == "deeplens_geolens_geometric"


def test_probe_completes_quickly():
    start = time.perf_counter()
    result = run_lightweight_backend_probe(backend_id="phase_to_fft_proxy")
    elapsed = time.perf_counter() - start
    assert elapsed < 10.0, f"Probe took {elapsed:.1f}s, expected <10s"
    assert result.status == "succeeded"


def test_probe_returns_psf_metrics():
    result = run_lightweight_backend_probe(backend_id="phase_to_fft_proxy")
    payload = result.result_payload or {}
    assert payload.get("psf_width_x", 0) >= 0
    assert payload.get("psf_energy", 0) > 0
    assert payload.get("differentiable") is True
