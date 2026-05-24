"""Claim Ceiling Resolver for Phase 41.

Computes the final claim ceiling as the minimum of:
- handler_claim_ceiling (from HandlerCapabilityRegistry)
- backend_claim_ceiling (from backend registry)
- dataset_claim_ceiling (synthetic vs real)
- execution_fidelity_claim_ceiling (lightweight_proxy vs native)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _evidence_rank(level: str) -> int:
    """Numeric rank for evidence level comparison. Higher = stronger claim allowed."""
    ranks: dict[str, int] = {
        "unsupported": 0,
        "structured_unsupported": 0,
        "needs_followup": 0,
        "requires_user_data": 0,
        "local_execution_completed": 6,
        "report_only": 1,
        "negative_result": 1,
        "mock_simulation": 2,
        "deeplens_integration_smoke": 3,
        "native_component_optimization": 4,
        "native_hsi_proxy": 5,
        "native_full_reconstruction_proxy": 6,
        "lightweight_scientific_execution": 7,
        "synthetic_lightweight_metric_experiment": 7,
        "synthetic_hsi_simulation": 7,
        "sweep_analysis": 7,
        "native_lens_simulation": 8,
        "native_waveoptics_simulation": 9,
        "native_waveoptics": 9,
        "stable_native_lens_hsi_codesign": 10,
        "rollback_protected_native_lens_hsi": 11,
        "real_hsi_performance": 12,
        "real_hsi": 12,
        "real_hsi_validation": 12,
    }
    return ranks.get(level, 0)


@dataclass
class ClaimCeilingResult:
    handler_id: str = ""
    design_backend_id: str = ""
    handler_claim_ceiling: str = "unsupported"
    backend_claim_ceiling: str = "unsupported"
    dataset_claim_ceiling: str = "unsupported"
    execution_fidelity_claim_ceiling: str = "unsupported"
    final_claim_ceiling: str = "unsupported"
    ceiling_source: str = "unknown"
    limiting_factor: str = ""
    downgrade_reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def resolve_claim_ceiling(
    handler_id: str = "",
    backend_id: str = "",
    dataset: str = "synthetic",
    execution_fidelity: str = "",
    evidence_level: str = "",
    synthetic_data: bool = False,
    physical_backend: bool | None = None,
    native_backend: bool | None = None,
    real_data: bool = False,
    proxy_fallback_used: bool = False,
    full_wave_optics: bool = False,
    phase_to_fft_proxy_used: bool | None = None,
) -> ClaimCeilingResult:
    """Resolve final claim ceiling from all constraints.

    Returns the most restrictive (lowest rank) ceiling across handler,
    backend, dataset, and execution fidelity.
    """
    result = ClaimCeilingResult(
        handler_id=handler_id,
        design_backend_id=backend_id,
    )
    ceilings: list[tuple[str, str]] = []  # (ceiling_value, source_name)

    # 1. Handler ceiling
    handler_ceiling = _get_handler_claim_ceiling(handler_id)
    result.handler_claim_ceiling = handler_ceiling or "unsupported"
    if handler_id and handler_ceiling == "needs_followup":
        result.warnings.append(
            f"Unknown handler_id '{handler_id}' — claim ceiling set to needs_followup"
        )
        result.final_claim_ceiling = "needs_followup"
        result.ceiling_source = "handler"
        result.limiting_factor = "unknown_handler_capability"
        result.downgrade_reasons.append("Handler capability not found in registry")
        return result
    if handler_ceiling:
        ceilings.append((handler_ceiling, "handler"))

    # 2. Backend ceiling
    backend_ceiling = _get_backend_claim_ceiling(backend_id)
    result.backend_claim_ceiling = backend_ceiling
    ceilings.append((backend_ceiling, "backend"))

    # 3. Dataset ceiling (only applied when explicitly flagged)
    if real_data:
        result.dataset_claim_ceiling = "real_hsi_performance"
        ceilings.append(("real_hsi_performance", "dataset"))
    elif synthetic_data:
        result.dataset_claim_ceiling = "lightweight_scientific_execution"
        ceilings.append(("lightweight_scientific_execution", "dataset"))

    # 4. Execution fidelity ceiling
    fidelity_ceiling = _fidelity_ceiling(execution_fidelity, full_wave_optics, bool(native_backend))
    result.execution_fidelity_claim_ceiling = fidelity_ceiling
    if fidelity_ceiling:
        ceilings.append((fidelity_ceiling, "execution_fidelity"))

    # 5. Physical/native constraints (only when explicitly provided)
    if physical_backend is False:
        ceilings.append(("lightweight_scientific_execution", "no_physical_backend"))
        result.downgrade_reasons.append("No physical backend — limited to lightweight")
    if native_backend is False:
        ceilings.append(("lightweight_scientific_execution", "no_native_backend"))
        result.downgrade_reasons.append("No native backend — limited to lightweight")
    if proxy_fallback_used:
        ceilings.append(("lightweight_scientific_execution", "proxy_fallback"))
        result.downgrade_reasons.append("Proxy fallback used — limited to lightweight")
    if phase_to_fft_proxy_used is True and not full_wave_optics:
        ceilings.append(("lightweight_scientific_execution", "fft_proxy_no_waveoptics"))
        result.downgrade_reasons.append("FFT proxy without wave-optics — limited to lightweight")

    # 6. Find minimum (most restrictive)
    if not ceilings:
        result.final_claim_ceiling = "unsupported"
        result.ceiling_source = "none"
        result.limiting_factor = "no_ceilings_computed"
        return result

    min_ceiling, min_source = min(ceilings, key=lambda x: _evidence_rank(x[0]))
    result.final_claim_ceiling = min_ceiling
    result.ceiling_source = min_source
    result.limiting_factor = f"Most restrictive ceiling from {min_source}: {min_ceiling}"

    # Add downgrade reasons for each ceiling lower than handler
    handler_rank = _evidence_rank(result.handler_claim_ceiling)
    for ceiling_val, source in ceilings:
        if _evidence_rank(ceiling_val) < handler_rank and source != "handler":
            result.downgrade_reasons.append(
                f"{source} ceiling ({ceiling_val}) lower than handler ceiling "
                f"({result.handler_claim_ceiling})"
            )

    return result


def _get_handler_claim_ceiling(handler_id: str) -> str:
    if not handler_id:
        return ""  # No handler — ceiling will come from other sources
    try:
        from optiresearch.skills.handler_capability_registry import (
            get_handler_capability_registry,
        )
        registry = get_handler_capability_registry()
        cap = registry.get(handler_id)
        if cap:
            return cap.max_claim_ceiling
    except Exception:
        pass
    return "needs_followup"


def _get_backend_claim_ceiling(backend_id: str) -> str:
    if not backend_id:
        return "unsupported"
    try:
        from optiresearch.backends.registry import get_backend
        backend = get_backend(backend_id)
        if backend is not None:
            return backend.claim_ceiling
    except Exception:
        pass
    return "unsupported"


def _fidelity_ceiling(
    execution_fidelity: str,
    full_wave_optics: bool,
    native_backend: bool,
) -> str:
    if not execution_fidelity:
        return ""
    fidelity_lower = execution_fidelity.lower()
    if "lightweight" in fidelity_lower or "proxy" in fidelity_lower:
        return "lightweight_scientific_execution"
    if "deeplens_native" in fidelity_lower or "native_geometric" in fidelity_lower:
        if full_wave_optics:
            return "native_waveoptics_simulation"
        return "native_lens_simulation"
    return ""
