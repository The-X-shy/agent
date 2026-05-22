"""DeepLens-specific backend helpers — registration of DeepLens backend variants."""

from __future__ import annotations

from optiresearch.backends.base import OpticalBackend
from optiresearch.backends.registry import register_backend


def register_deeplens_backends() -> list[OpticalBackend]:
    """Register all DeepLens variant backends (idempotent).

    The core 8 backends are already registered in registry._seed_registry().
    This function is a convenience for explicitly re-registering or overriding
    DeepLens backends with updated metadata.
    """
    backends: list[OpticalBackend] = [
        OpticalBackend(
            backend_id="deeplens_fresnel_component",
            label="DeepLens Fresnel Diffractive Surface Component",
            backend_type="deeplens",
            differentiability_level="native_component",
            supports_psf_generation=True,
            supports_native_optimization=True,
            requires_lens_file=True,
            claim_ceiling="native_component_optimization",
            known_failure_modes=[
                "FFT proxy, not full wave propagation",
                "single-surface only — no compound lens effects",
            ],
            recommended_use_cases=[
                "Fresnel DOE surface optimization",
                "component-level gradient validation",
            ],
        ),
        OpticalBackend(
            backend_id="deeplens_binary2phase_component",
            label="DeepLens Binary2Phase Diffractive Surface Component",
            backend_type="deeplens",
            differentiability_level="native_component",
            supports_psf_generation=True,
            supports_native_optimization=True,
            requires_lens_file=True,
            claim_ceiling="native_component_optimization",
            known_failure_modes=[
                "FFT proxy, not full wave propagation",
                "binary quantization may create gradient discontinuities",
            ],
            recommended_use_cases=[
                "Binary2Phase DOE surface optimization",
                "component-level gradient validation",
            ],
        ),
        OpticalBackend(
            backend_id="deeplens_geolens_geometric",
            label="DeepLens GeoLens Geometric Ray-Tracing PSF",
            backend_type="deeplens",
            differentiability_level="native_lens_simulation",
            supports_psf_generation=True,
            supports_image_simulation=True,
            supports_hsi_forward=True,
            supports_native_optimization=True,
            requires_lens_file=True,
            supports_remote_execution=True,
            claim_ceiling="native_lens_simulation",
            known_failure_modes=[
                "geometric ray-tracing only — not coherent wave-optics",
                "optical gradients can exceed 1700 at default LR",
                "large gradients cause HSI reconstruction loss to spike",
            ],
            recommended_use_cases=[
                "native lens simulation HSI co-design",
                "stability training with rollback",
                "remote WSL validation",
            ],
        ),
        OpticalBackend(
            backend_id="deeplens_coherent_asm",
            label="DeepLens GeoLens Coherent ASM Wave Path",
            backend_type="deeplens",
            differentiability_level="none",
            supports_psf_generation=True,
            requires_lens_file=True,
            supports_remote_execution=True,
            claim_ceiling="native_lens_simulation",
            known_failure_modes=[
                "coherent ASM ray sampling uses no_grad / breaks autograd",
                "psf.requires_grad is False — cannot support native_waveoptics claim",
            ],
            recommended_use_cases=[
                "wave-optics path inspection",
                "non-differentiable wave propagation probe",
            ],
        ),
        OpticalBackend(
            backend_id="deeplens_blackbox_source_psf",
            label="DeepLens Black-Box Source PSF",
            backend_type="deeplens",
            differentiability_level="black_box",
            supports_psf_generation=True,
            supports_hsi_forward=True,
            requires_lens_file=True,
            supports_remote_execution=True,
            claim_ceiling="deeplens_integration_smoke",
            known_failure_modes=[
                "black-box — no gradient flow into optical parameters",
                "cannot support native gradient or optimization claims",
            ],
            recommended_use_cases=[
                "DeepLens integration smoke testing",
                "PSF quality baseline",
            ],
        ),
    ]

    for b in backends:
        register_backend(b)

    return backends
