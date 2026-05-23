"""Phase 32: Deep GeoLens geometric deep probe tests."""

import time
from optiresearch.runtime.lightweight_experiments import (
    run_deeplens_geolens_geometric_deep_probe,
)


def test_deep_probe_returns_structured_result():
    result = run_deeplens_geolens_geometric_deep_probe(
        backend_id="deeplens_geolens_geometric",
    )
    assert result.status in ("succeeded", "failed")
    assert result.result_payload is not None
    assert "probe_depth" in result.result_payload
    assert result.result_payload.get("probe_depth") == "deep"


def test_deep_probe_never_crashes():
    result = run_deeplens_geolens_geometric_deep_probe(
        backend_id="deeplens_geolens_geometric",
    )
    assert result.result_payload is not None
    assert "probe_time_seconds" in result.result_payload
    assert "deeplens_available" in result.result_payload


def test_deep_probe_completes_quickly():
    start = time.perf_counter()
    result = run_deeplens_geolens_geometric_deep_probe(
        backend_id="deeplens_geolens_geometric",
    )
    elapsed = time.perf_counter() - start
    assert elapsed < 30.0, f"Deep probe took {elapsed:.1f}s, expected <30s"


def test_deep_probe_reports_correct_fields():
    result = run_deeplens_geolens_geometric_deep_probe(
        backend_id="deeplens_geolens_geometric",
    )
    payload = result.result_payload or {}
    assert "probe_depth" in payload
    assert "deeplens_available" in payload
    if result.status == "succeeded":
        assert payload.get("full_wave_optics") is False
        assert payload.get("phase_to_fft_proxy_used") is False
        assert payload.get("deeplens_native_psf_path") == "geolens.psf_geometric"
        assert "optical_gradient_norm" in payload
        assert "parameters_changed" in payload
    else:
        assert "error_code" in payload
