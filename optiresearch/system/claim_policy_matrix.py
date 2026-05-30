"""Generate claim policy matrix from evidence level registry (Phase 68)."""
from __future__ import annotations

from typing import Any


def generate_claim_policy_matrix() -> dict[str, Any]:
    """Generate matrix with evidence levels and their claim policies.

    Derived from existing sources:
    - claim_gate_v2.py: violation types and evidence rank ladder
    - handler_capability_registry.py: handler evidence levels
    - backends/registry.py: backend claim ceilings
    """
    from optiresearch.memory.claim_gate_v2 import _evidence_rank

    rows: list[dict[str, Any]] = []
    evidence_levels = [
        "unsupported",
        "report_only",
        "negative_result",
        "mock_simulation",
        "deeplens_integration_smoke",
        "native_component_optimization",
        "component_surrogate_hsi_codesign",
        "native_hsi_proxy",
        "native_full_reconstruction_proxy",
        "lightweight_scientific_execution",
        "sweep_analysis",
        "native_lens_simulation",
        "native_waveoptics_simulation",
        "stable_native_lens_hsi_codesign",
        "rollback_protected_native_lens_hsi",
        "real_hsi_performance",
    ]

    _safe_wording = {
        "unsupported": "This capability is not currently supported.",
        "report_only": "This finding is documented for traceability only — no scientific claim is made.",
        "negative_result": "Negative result documented — no positive performance claim is made.",
        "mock_simulation": "Results are from mock simulation only — not reflective of real hardware.",
        "deeplens_integration_smoke": "DeepLens integration verified at smoke-test level only.",
        "native_component_optimization": "Component-level optimization result — not full lens validation.",
        "component_surrogate_hsi_codesign": "Component surrogate HSI co-design result — synthetic data only.",
        "native_hsi_proxy": "Native HSI proxy result — not full reconstruction validation.",
        "native_full_reconstruction_proxy": "Full reconstruction proxy result — not wave-optics validated.",
        "lightweight_scientific_execution": "Lightweight scientific execution — limited metric scope.",
        "sweep_analysis": "Parameter sweep analysis — identifies trends, not optimal designs.",
        "native_lens_simulation": "Native lens simulation result — geometric optics only.",
        "native_waveoptics_simulation": "Native wave-optics simulation — highest fidelity simulation.",
        "stable_native_lens_hsi_codesign": "Stable native lens HSI co-design — reproducibility validated.",
        "rollback_protected_native_lens_hsi": "Rollback-protected native lens HSI — loss-stability verified.",
        "real_hsi_performance": "Real HSI performance validated on physical measurements.",
    }

    _blocked_claims = {
        "unsupported": ["Any scientific or performance claim"],
        "report_only": ["Performance improvement", "Optical quality improvement"],
        "negative_result": ["Positive performance claim", "Optical improvement"],
        "mock_simulation": ["Real hardware performance", "Native DeepLens performance"],
        "deeplens_integration_smoke": ["Optimization quality", "Optical improvement"],
        "native_component_optimization": ["Full lens optimization", "Real HSI performance", "Wave-optics validation"],
        "component_surrogate_hsi_codesign": ["Full GeoLens performance", "Real camera validation", "Wave-optics", "Real HSI performance"],
        "native_hsi_proxy": ["Wave-optics validation", "Real HSI performance"],
        "native_full_reconstruction_proxy": ["Wave-optics", "Coherent propagation", "Real HSI"],
        "lightweight_scientific_execution": ["Real HSI performance", "Native lens validation"],
        "sweep_analysis": ["Optimal design found", "Real HSI performance"],
        "native_lens_simulation": ["Wave-optics validation", "Real HSI performance", "Coherent propagation"],
        "native_waveoptics_simulation": ["Real HSI performance", "Physical measurement validation"],
        "stable_native_lens_hsi_codesign": ["Real HSI performance", "Physical camera validation"],
        "rollback_protected_native_lens_hsi": ["Real HSI performance"],
        "real_hsi_performance": [],
    }

    for level in evidence_levels:
        rank = _evidence_rank(level)
        rows.append({
            "evidence_level": level,
            "rank": rank,
            "source_backend": "any" if rank <= 2 else ("deeplens" if rank >= 4 else "proxy"),
            "dataset_type": "real" if rank >= 12 else "synthetic",
            "execution_fidelity": "full" if rank >= 8 else ("partial" if rank >= 4 else "minimal"),
            "supported_claims": _supported_claims(level, rank),
            "blocked_claims": _blocked_claims.get(level, []),
            "safe_wording_template": _safe_wording.get(level, ""),
            "required_artifacts": _required_artifacts(rank),
            "required_metrics": _required_metrics(rank),
            "downgrade_conditions": _downgrade_conditions(level, rank),
        })

    return {
        "matrix_version": "0.1",
        "evidence_levels_covered": len(evidence_levels),
        "rows": rows,
    }


def _supported_claims(level: str, rank: int) -> list[str]:
    if rank <= 0:
        return []
    if rank <= 1:
        return ["Documentation and traceability"]
    if rank <= 2:
        return ["Mock pipeline integration correctness"]
    if rank <= 3:
        return ["DeepLens API integration verified"]
    if rank <= 4:
        return ["Component-level gradient optimization possible"]
    if rank <= 5:
        return ["Component surrogate HSI co-design synthetic performance"]
    if rank <= 7:
        return ["Synthetic lightweight HSI co-design results"]
    if rank <= 8:
        return ["Native lens simulation synthetic HSI performance"]
    if rank <= 9:
        return ["Native wave-optics simulation synthetic HSI performance"]
    if rank <= 10:
        return ["Stable, reproducible native lens HSI optimization"]
    if rank <= 11:
        return ["Rollback-protected, stability-verified native lens HSI optimization"]
    return ["Real HSI performance validated on physical measurements"]


def _required_artifacts(rank: int) -> list[str]:
    base = ["result.json"]
    if rank >= 3:
        base.append("metrics.json")
    if rank >= 5:
        base.append("artifact_manifest.json")
    if rank >= 7:
        base.append("report.md")
    if rank >= 10:
        base.extend(["stability_trace.json", "benchmark_summary.json"])
    if rank >= 12:
        base.append("real_camera_validation.json")
    return base


def _required_metrics(rank: int) -> list[str]:
    if rank <= 1:
        return []
    if rank <= 3:
        return ["api_call_success"]
    if rank <= 5:
        return ["mse", "psnr"]
    if rank <= 7:
        return ["mse", "psnr", "sam"]
    if rank <= 9:
        return ["mse", "psnr", "sam", "grad_norm_max"]
    return ["mse", "psnr", "sam", "grad_norm_max", "stability_score", "rollback_count", "completion_rate"]


def _downgrade_conditions(level: str, rank: int) -> list[str]:
    conditions = []
    if rank >= 8:
        conditions.append("If rollback triggered, downgrade to lightweight_scientific_execution")
    if rank >= 10:
        conditions.append("If completion_rate < 50%, downgrade to native_lens_simulation")
    if rank >= 5:
        conditions.append("If component not differentiable, downgrade to deeplens_integration_smoke")
    return conditions
