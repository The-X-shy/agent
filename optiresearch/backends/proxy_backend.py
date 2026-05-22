"""Proxy backend helpers — registration of proxy and synthetic backends."""

from __future__ import annotations

from optiresearch.backends.base import OpticalBackend
from optiresearch.backends.registry import register_backend


def register_proxy_backends() -> list[OpticalBackend]:
    """Register proxy and synthetic backend variants (idempotent).

    The core 8 backends are already registered in registry._seed_registry().
    This function is a convenience for explicitly re-registering or overriding
    proxy backends with updated metadata.
    """
    backends: list[OpticalBackend] = [
        OpticalBackend(
            backend_id="phase_to_fft_proxy",
            label="Phase-to-FFT Differentiable Proxy",
            backend_type="proxy",
            differentiability_level="differentiable_proxy",
            supports_psf_generation=True,
            supports_image_simulation=True,
            supports_hsi_forward=True,
            supports_native_optimization=True,
            claim_ceiling="native_full_reconstruction_proxy",
            known_failure_modes=[
                "not full wave-optics — FFT of scalar phase is a proxy",
                "cannot support native_waveoptics or coherent claims",
            ],
            recommended_use_cases=[
                "differentiable HSI proxy co-design",
                "component-level differentiable optimization",
            ],
        ),
        OpticalBackend(
            backend_id="mock_deeplens",
            label="Mock DeepLens (deterministic numpy PSF)",
            backend_type="mock",
            differentiability_level="none",
            supports_psf_generation=True,
            supports_image_simulation=True,
            supports_hsi_forward=True,
            claim_ceiling="mock_simulation",
            known_failure_modes=[
                "no gradient flow",
                "not reflective of real hardware",
            ],
            recommended_use_cases=[
                "baseline comparison",
                "pipeline integration testing",
            ],
        ),
        OpticalBackend(
            backend_id="local_synthetic_hsi",
            label="Local Synthetic HSI Backend",
            backend_type="synthetic",
            differentiability_level="differentiable_proxy",
            supports_psf_generation=True,
            supports_image_simulation=True,
            supports_hsi_forward=True,
            supports_native_optimization=True,
            claim_ceiling="synthetic_hsi_simulation",
            known_failure_modes=[
                "synthetic data — cannot support real HSI performance claims",
                "no real sensor noise or calibration effects",
            ],
            recommended_use_cases=[
                "local algorithm development",
                "differentiable pipeline testing",
                "synthetic HSI reconstruction baseline",
            ],
        ),
    ]

    for b in backends:
        register_backend(b)

    return backends
