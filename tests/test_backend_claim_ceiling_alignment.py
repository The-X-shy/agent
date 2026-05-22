"""Test that backend claim ceilings correctly cap task evidence levels."""

import pytest
from optiresearch.backends.registry import get_backend_task_evidence_cap


def test_phase_to_fft_proxy_allows_stable_lens_hsi_capped():
    cap = get_backend_task_evidence_cap("phase_to_fft_proxy", "stable_lens_hsi_codesign")
    assert cap == "native_full_reconstruction_proxy"


def test_phase_to_fft_proxy_rejects_native_waveoptics():
    cap = get_backend_task_evidence_cap("phase_to_fft_proxy", "native_waveoptics_codesign")
    assert cap is None


def test_geolens_allows_psf_probe():
    cap = get_backend_task_evidence_cap("deeplens_geolens_geometric", "psf_probe")
    assert cap == "deeplens_integration_smoke"


def test_geolens_allows_stable_lens_hsi_at_native_lens_simulation():
    cap = get_backend_task_evidence_cap("deeplens_geolens_geometric", "stable_lens_hsi_codesign")
    assert cap == "native_lens_simulation"


def test_unknown_backend_returns_none():
    cap = get_backend_task_evidence_cap("nonexistent", "stable_lens_hsi_codesign")
    assert cap is None


def test_phase_to_fft_proxy_has_multiple_task_types():
    cap_hsi = get_backend_task_evidence_cap("phase_to_fft_proxy", "native_hsi_codesign")
    cap_recon = get_backend_task_evidence_cap("phase_to_fft_proxy", "native_hsi_reconstruction_codesign")
    cap_psf = get_backend_task_evidence_cap("phase_to_fft_proxy", "lightweight_psf_probe")
    assert cap_hsi == "native_hsi_proxy"
    assert cap_recon == "native_full_reconstruction_proxy"
    assert cap_psf == "deeplens_integration_smoke"


def test_geolens_rejects_lightweight_psf_probe():
    cap = get_backend_task_evidence_cap("deeplens_geolens_geometric", "lightweight_psf_probe")
    assert cap is None


def test_mock_deeplens_allows_lightweight_tasks():
    cap = get_backend_task_evidence_cap("mock_deeplens", "lightweight_psf_probe")
    assert cap == "mock_simulation"


def test_local_synthetic_allows_stable_lens_hsi():
    cap = get_backend_task_evidence_cap("local_synthetic_hsi", "stable_lens_hsi_codesign")
    assert cap == "synthetic_hsi_simulation"
