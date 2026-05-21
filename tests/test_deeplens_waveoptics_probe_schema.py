"""Tests for Phase 22 wave-optics probe schema."""

import pytest
from optiresearch.schemas.deeplens_waveoptics_probe import (
    DeepLensWaveOpticsProbeResult,
    DeepLensWaveOpticsProbeSpec,
    make_waveoptics_probe_id,
)


def test_spec_create():
    spec = DeepLensWaveOpticsProbeSpec(
        run_id=make_waveoptics_probe_id("GeoLensCooke", "minimize_psf_width"),
        candidate="GeoLensCooke",
        objective="minimize_psf_width",
    )
    assert spec.candidate == "GeoLensCooke"
    assert spec.strict_waveoptics is True
    assert spec.allow_phase_to_fft_proxy is False


def test_spec_rejects_invalid_candidate():
    with pytest.raises(ValueError):
        DeepLensWaveOpticsProbeSpec(run_id="x", candidate="bad", objective="minimize_psf_width")


def test_spec_rejects_invalid_objective():
    with pytest.raises(ValueError):
        DeepLensWaveOpticsProbeSpec(run_id="x", candidate="GeoLensCooke", objective="bad")


def test_result_waveoptics_fields():
    r = DeepLensWaveOpticsProbeResult(
        run_id="x", status="unsupported", candidate="GeoLensCooke",
    )
    assert r.full_wave_optics is False
    assert r.phase_to_fft_proxy_used is True
    assert r.psf_requires_grad is False


def test_result_succeeded():
    r = DeepLensWaveOpticsProbeResult(
        run_id="x", status="succeeded", candidate="GeoLensCooke",
        full_wave_optics=False, phase_to_fft_proxy_used=False, differentiable=True,
        optical_gradient_norm=0.14, optical_parameters_changed=True,
        psf_requires_grad=True, autograd_graph_exists=True,
        deeplens_native_wave_path="geolens.psf_geometric",
        evidence_level="native_lens_simulation",
    )
    assert r.status == "succeeded"
    assert r.evidence_level == "native_lens_simulation"


def test_make_id_deterministic():
    a = make_waveoptics_probe_id("GeoLensCooke", "minimize_psf_width")
    b = make_waveoptics_probe_id("GeoLensCooke", "minimize_psf_width")
    assert a == b
