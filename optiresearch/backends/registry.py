"""Backend registry with 8 registered optical backends.

This module has zero project imports — it is a pure-data anchor
that every other Phase 24 component can safely import.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from optiresearch.backends.base import OpticalBackend

_registry: dict[str, OpticalBackend] = {}


def _seed_registry() -> None:
    """Register all 8 known backends. Idempotent — call freely."""
    if _registry:
        return

    backends: list[OpticalBackend] = [
        OpticalBackend(
            backend_id="mock_deeplens",
            label="Mock DeepLens (deterministic numpy PSF)",
            backend_type="mock",
            differentiability_level="none",
            supports_psf_generation=True,
            supports_image_simulation=True,
            supports_hsi_forward=True,
            claim_ceiling="mock_simulation",
            known_failure_modes=["no gradient flow", "not reflective of real hardware"],
            recommended_use_cases=["baseline comparison", "pipeline integration testing"],
        ),
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
                "requires DiffractiveLens or pure wave propagation, not GeoLens ASM",
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
        _registry[b.backend_id] = b


def get_backend_registry() -> dict[str, OpticalBackend]:
    """Return the full backend registry (id -> OpticalBackend)."""
    _seed_registry()
    return dict(_registry)


def register_backend(backend: OpticalBackend) -> None:
    """Register a new or override an existing backend."""
    _seed_registry()
    _registry[backend.backend_id] = backend


def list_backends() -> list[OpticalBackend]:
    """Return all registered backends as a list."""
    _seed_registry()
    return list(_registry.values())


def get_backend(backend_id: str) -> Optional[OpticalBackend]:
    """Look up a single backend by id."""
    _seed_registry()
    return _registry.get(backend_id)


def get_backend_by_claim_ceiling(ceiling: str) -> list[OpticalBackend]:
    """Find all backends with a given claim ceiling."""
    _seed_registry()
    return [b for b in _registry.values() if b.claim_ceiling == ceiling]


def export_backend_registry_markdown(path: Path) -> Path:
    """Write backend registry as a markdown table."""
    _seed_registry()
    lines = [
        "# Optical Backend Registry",
        "",
        f"| backend_id | type | diff_level | claim_ceiling |",
        "|---|---|---|---|",
    ]
    for b in sorted(_registry.values(), key=lambda x: x.backend_id):
        lines.append(
            f"| {b.backend_id} | {b.backend_type} | {b.differentiability_level} | {b.claim_ceiling} |"
        )
    lines.append("")
    lines.append("## Backend Details")
    lines.append("")
    for b in sorted(_registry.values(), key=lambda x: x.backend_id):
        lines.append(f"### {b.backend_id}")
        lines.append(f"- **Label:** {b.label}")
        lines.append(f"- **Type:** {b.backend_type}")
        lines.append(f"- **Differentiability:** {b.differentiability_level}")
        lines.append(f"- **Claim Ceiling:** {b.claim_ceiling}")
        if b.known_failure_modes:
            lines.append("- **Known Failure Modes:**")
            for fm in b.known_failure_modes:
                lines.append(f"  - {fm}")
        if b.recommended_use_cases:
            lines.append("- **Recommended Use Cases:**")
            for uc in b.recommended_use_cases:
                lines.append(f"  - {uc}")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# Backend -> allowed task types with evidence_level_cap.
# If a task_type is not listed for a backend, it is not allowed on that backend.
_BACKEND_TASK_RULES: dict[str, dict[str, Optional[str]]] = {
    "phase_to_fft_proxy": {
        "stable_lens_hsi_codesign": "native_full_reconstruction_proxy",
        "native_hsi_codesign": "native_hsi_proxy",
        "native_hsi_reconstruction_codesign": "native_full_reconstruction_proxy",
        "lightweight_psf_probe": "deeplens_integration_smoke",
    },
    "deeplens_geolens_geometric": {
        "stable_lens_hsi_codesign": "native_lens_simulation",
        "native_hsi_codesign": "native_hsi_proxy",
        "native_hsi_reconstruction_codesign": "native_full_reconstruction_proxy",
        "psf_probe": "deeplens_integration_smoke",
        "native_optimization_probe": "native_component_optimization",
    },
    "mock_deeplens": {
        "stable_lens_hsi_codesign": "mock_simulation",
        "native_hsi_codesign": "mock_simulation",
        "lightweight_psf_probe": "mock_simulation",
    },
    "local_synthetic_hsi": {
        "stable_lens_hsi_codesign": "synthetic_hsi_simulation",
        "native_hsi_codesign": "synthetic_hsi_simulation",
        "native_hsi_reconstruction_codesign": "synthetic_hsi_simulation",
        "lightweight_psf_probe": "synthetic_hsi_simulation",
    },
    "deeplens_fresnel_component": {
        "native_optimization_probe": "native_component_optimization",
        "component_optimization": "native_component_optimization",
    },
    "deeplens_binary2phase_component": {
        "native_optimization_probe": "native_component_optimization",
        "component_optimization": "native_component_optimization",
    },
    "deeplens_coherent_asm": {
        "psf_probe": "deeplens_integration_smoke",
        "lightweight_psf_probe": "deeplens_integration_smoke",
    },
    "deeplens_blackbox_source_psf": {
        "psf_probe": "deeplens_integration_smoke",
        "lightweight_psf_probe": "deeplens_integration_smoke",
    },
}


def get_backend_task_evidence_cap(backend_id: str, task_type: str) -> Optional[str]:
    """Get the evidence level cap for a task on a specific backend.

    Returns None if the task is not allowed on this backend.
    """
    rules = _BACKEND_TASK_RULES.get(backend_id, {})
    return rules.get(task_type)


def export_backend_registry_json(path: Path) -> Path:
    """Write backend registry as a JSON file."""
    _seed_registry()
    payload = {
        bid: {
            "backend_id": b.backend_id,
            "label": b.label,
            "backend_type": b.backend_type,
            "differentiability_level": b.differentiability_level,
            "supports_psf_generation": b.supports_psf_generation,
            "supports_native_optimization": b.supports_native_optimization,
            "supports_full_waveoptics": b.supports_full_waveoptics,
            "claim_ceiling": b.claim_ceiling,
            "known_failure_modes": b.known_failure_modes,
            "recommended_use_cases": b.recommended_use_cases,
        }
        for bid, b in sorted(_registry.items())
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
