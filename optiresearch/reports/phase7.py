"""Phase 7 DeepLens encoder-proxy experiment report."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from optiresearch.adapters.deeplens import DeepLensAdapter
from optiresearch.adapters.deeplens_encoder_strategies import list_deeplens_encoder_strategies
from optiresearch.reports.backend_alignment import compare_backend_metrics, load_backend_baseline


def report_root() -> Path:
    return Path(os.getenv("OPTIRESEARCH_REPORT_ROOT", "./workspace/reports"))


def export_phase7_report() -> Path:
    root = report_root()
    root.mkdir(parents=True, exist_ok=True)
    path = root / "phase7_deeplens_encoder_proxy_report.md"
    path.write_text(_markdown(), encoding="utf-8")
    return path


def _markdown() -> str:
    environment = DeepLensAdapter().validate_environment()
    mock = load_backend_baseline("mock_deeplens")
    real = load_backend_baseline("deeplens")
    alignment = compare_backend_metrics("mock_deeplens", "deeplens")
    objective = real.get("objective") or mock.get("objective") or "DeepLens encoder-specific proxy baseline"
    lines = [
        "# Phase 7 DeepLens Encoder Proxy Report",
        "",
        "## Objective",
        "",
        str(objective),
        "",
        "## DeepLens environment",
        "",
        f"DeepLens available: `{environment.get('available')}`",
        f"DeepLens version: `{environment.get('deeplens_version')}`",
        f"Python version: `{environment.get('python_version')}`",
        "",
        "## Encoder strategy registry",
        "",
        "| Encoder | Strategy | Realization Level | Description |",
        "|---|---|---|---|",
    ]
    for strategy in list_deeplens_encoder_strategies():
        lines.append(
            f"| {strategy.encoder_type} | {strategy.strategy_name} | {strategy.realization_level} | {strategy.description} |"
        )
    lines.extend(
        [
            "",
            "## Realization level summary",
            "",
            "Phase 7 uses real DeepLens base PSF generation plus adapter-level encoder proxy transformation. It is not native physical encoder optimization.",
            "",
            "## DeepLens encoder baseline table",
            "",
            "| Encoder | Joint | Depth | Spectral | Realization | Physical Validation | Proxy Transform |",
            "|---|---:|---:|---:|---|---|---|",
        ]
    )
    for item in real.get("runs", []):
        metrics = item.get("metrics", {})
        lines.append(
            "| {encoder} | {joint} | {depth} | {spectral} | {level} | {physical} | {proxy} |".format(
                encoder=item.get("encoder_type"),
                joint=item.get("joint_tradeoff_score"),
                depth=metrics.get("psf_depth_similarity"),
                spectral=metrics.get("spectral_separability"),
                level=metrics.get("encoder_behavior_realization_level"),
                physical=metrics.get("physical_validation_level"),
                proxy=metrics.get("proxy_transform_name"),
            )
        )
    if not real.get("runs"):
        lines.append("| missing |  |  |  |  |  |  |")
    lines.extend(
        [
            "",
            "## Mock-real rank alignment",
            "",
            f"Simple pairwise rank agreement: `{alignment['summary']['rank_agreement']}`",
            "",
            "## Evidence level and claim caveats",
            "",
            "- Supported: adapter-level encoder behavior claims under adapter-proxy DeepLens setting.",
            "- Not supported: native physical encoder optimization claims.",
            "- Caveat: adapter-proxy DeepLens evidence; not native physical validation.",
            "",
            "## What is validated",
            "",
            "- Real DeepLens base PSF generation can feed the OptiResearch artifact, memory, and evidence pipeline.",
            "- The adapter can produce encoder-specific baseline artifacts through explicit proxy transforms.",
            "- Baseline comparison and mock-real rank alignment reports can be exported.",
            "",
            "## What is not validated",
            "",
            "- Native DeepLens physical encoder designs for the five encoder families.",
            "- Full EDOF-HSI wavelength-aware optical optimization.",
            "- Final optical performance claims.",
            "",
            "## Requirements before full EDOF-HSI optimization",
            "",
            "1. Map each encoder family to native DeepLens optical elements or phase masks.",
            "2. Replace proxy transforms with differentiable optical simulation and optimization.",
            "3. Validate HSI wavelength behavior against physical design parameters.",
            "4. Promote design rules only after native or experimentally validated evidence is available.",
            "",
        ]
    )
    return "\n".join(lines)
