"""Phase 8 semi-native DeepLens and LLM integration report."""

from __future__ import annotations

import os
from pathlib import Path

from optiresearch.adapters.deeplens import DeepLensAdapter
from optiresearch.adapters.deeplens_api_probe import probe_deeplens_api
from optiresearch.adapters.deeplens_encoder_strategies import list_deeplens_encoder_strategies
from optiresearch.reports.backend_alignment import load_backend_baseline


def export_phase8_report() -> Path:
    root = Path(os.getenv("OPTIRESEARCH_REPORT_ROOT", "./workspace/reports"))
    root.mkdir(parents=True, exist_ok=True)
    path = root / "phase8_deeplens_semi_native_report.md"
    path.write_text(_markdown(), encoding="utf-8")
    return path


def _markdown() -> str:
    probe = probe_deeplens_api()
    env = DeepLensAdapter().validate_environment()
    baseline = load_backend_baseline("deeplens")
    lines = [
        "# Phase 8 DeepLens Semi-Native Report",
        "",
        "## Objective",
        "",
        str(baseline.get("objective", "DeepLens semi-native encoder protocol")),
        "",
        "## DeepLens API probe summary",
        "",
        f"Available: `{probe.get('available')}`",
        f"Lens candidates: `{len(probe.get('candidate_lens_classes', []))}`",
        f"Surface candidates: `{len(probe.get('candidate_surface_classes', []))}`",
        f"Phase/DOE candidates: `{len(probe.get('candidate_phase_or_doe_classes', []))}`",
        "",
        "## Capability model",
        "",
        "| Capability | Available |",
        "|---|---|",
    ]
    for item in env.get("capabilities", []):
        lines.append(f"| {item['name']} | {item['available']} |")
    lines.extend(
        [
            "",
            "## Encoder strategy table",
            "",
            "| Encoder | Default Level | Claim Scope | Semi-native Plan |",
            "|---|---|---|---|",
        ]
    )
    for strategy in list_deeplens_encoder_strategies():
        lines.append(
            f"| {strategy.encoder_type} | {strategy.realization_level} | {strategy.claim_scope} | {'; '.join(strategy.semi_native_plan)} |"
        )
    lines.extend(
        [
            "",
            "## Realization level table",
            "",
            "| Encoder | Selected | Semi-native Succeeded | Proxy Fallback | Claim Scope | Joint |",
            "|---|---|---|---|---|---:|",
        ]
    )
    for item in baseline.get("runs", []):
        metrics = item.get("metrics", {})
        lines.append(
            "| {encoder} | {selected} | {succeeded} | {fallback} | {scope} | {joint} |".format(
                encoder=item.get("encoder_type"),
                selected=metrics.get("selected_realization_level"),
                succeeded=metrics.get("semi_native_succeeded"),
                fallback=metrics.get("proxy_fallback_used"),
                scope=metrics.get("claim_scope"),
                joint=item.get("joint_tradeoff_score"),
            )
        )
    lines.extend(
        [
            "",
            "## Baseline comparison under selected realization",
            "",
            "See `workspace/baselines/deeplens/baseline_comparison.md`.",
            "",
            "## Adapter-proxy vs semi-native distinction",
            "",
            "Semi-native means part of the behavior uses a DeepLens native lens-side or PSF generation mechanism. It is not native optimized optical design. Adapter-proxy means encoder behavior is represented outside the native DeepLens design path.",
            "",
            "## Claim scopes allowed and disallowed",
            "",
            "- Allowed: baseline DeepLens ParaxialLens behavior and adapter-proxy setting claims with explicit caveats.",
            "- Disallowed: native physical optimization, final optical validation, and full HSI pipeline claims.",
            "",
            "## Optimization readiness",
            "",
            "OptimizationSpec is draft-only. DeepLensAdapter returns `OPTIMIZATION_NOT_AVAILABLE` until native optimization is bound.",
            "",
            "## Requirements before native EDOF-HSI",
            "",
            "1. Bind native DeepLens phase/surface/DOE objects.",
            "2. Add wavelength-aware HSI PSF simulation.",
            "3. Implement native optimization and reconstruction network evaluation.",
            "",
        ]
    )
    return "\n".join(lines)
