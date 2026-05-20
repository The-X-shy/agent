"""Phase 6 real DeepLens experiment report export."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from optiresearch.adapters.deeplens import DeepLensAdapter
from optiresearch.reports.backend_alignment import compare_backend_metrics, export_backend_alignment_report
from optiresearch.storage.sqlite_store import SQLiteStore


def export_phase6_report() -> Path:
    root = Path(os.getenv("OPTIRESEARCH_REPORT_ROOT", "./workspace/reports"))
    root.mkdir(parents=True, exist_ok=True)
    path = root / "phase6_real_deeplens_report.md"
    path.write_text(_markdown(), encoding="utf-8")
    return path


def _markdown() -> str:
    env = DeepLensAdapter().validate_environment()
    mock = _load_baseline("mock_deeplens")
    real = _load_baseline("deeplens")
    alignment = compare_backend_metrics("mock_deeplens", "deeplens")
    export_backend_alignment_report("mock_deeplens", "deeplens")
    claims = SQLiteStore().list("claims")
    lines = [
        "# Phase 6 Real DeepLens Report",
        "",
        "## Current Backend Status",
        "",
        f"DeepLens available: `{env.get('available')}`",
        f"DeepLens version: `{env.get('deeplens_version')}`",
        f"Python version: `{env.get('python_version')}`",
        "",
        "## DeepLens Environment",
        "",
        "| Capability | Available | Reason | Evidence |",
        "|---|---|---|---|",
    ]
    for item in env.get("capabilities", []):
        lines.append(f"| {item['name']} | {item['available']} | {item['reason']} | {item['evidence']} |")
    lines.extend(
        [
            "",
            "## Smoke Run Result",
            "",
            "Smoke-level DeepLens output validates adapter integration and standard artifact generation when DeepLens is installed.",
            "",
            "## Mock Baseline Result",
            "",
            _baseline_summary(mock),
            "",
            "## DeepLens Baseline Result",
            "",
            _baseline_summary(real),
            "",
            "## Mock-Real Alignment",
            "",
            f"Compared encoders: `{alignment['summary']['encoder_count']}`",
            "",
            "## Claim Caveats",
            "",
        ]
    )
    deeplens_claims = [claim for claim in claims if claim.get("scope", {}).get("backend") == "deeplens"]
    if deeplens_claims:
        for claim in deeplens_claims[-8:]:
            caveats = "; ".join(claim.get("required_caveats", []))
            lines.append(f"- {claim.get('status')}: {claim.get('text')} ({caveats})")
    else:
        lines.append("- Warning: no DeepLens claims found in the current store.")
    lines.extend(
        [
            "",
            "## What Is Validated",
            "",
            "- DeepLens import and environment detection.",
            "- ParaxialLens smoke PSF generation.",
            "- Standard artifact export and registration.",
            "- RunMemory and ClaimEvidence wiring for real-backend runs.",
            "",
            "## What Is Not Yet Validated",
            "",
            "- Encoder-specific optical behavior for EDOF-HSI families.",
            "- Wavelength-aware HSI PSF behavior beyond replicated smoke bands.",
            "- Real optimization-backed design rules.",
            "",
            "## Next Experiment Protocol",
            "",
            "1. Define concrete DeepLens lens configurations for each encoder family.",
            "2. Add wavelength-dependent simulation instead of smoke-level replication.",
            "3. Run matched mock and real baselines after encoder behavior is realized.",
            "4. Promote design rules only after full-backend evidence exists.",
            "",
        ]
    )
    if real.get("warning"):
        lines.append(f"Warning: {real['warning']}")
    return "\n".join(lines)


def _load_baseline(backend: str) -> dict[str, Any]:
    path = Path(os.getenv("OPTIRESEARCH_BASELINE_ROOT", "./workspace/baselines")) / backend / "baseline_comparison.json"
    if not path.exists():
        return {"backend": backend, "runs": [], "warning": f"missing baseline: {path}"}
    return json.loads(path.read_text(encoding="utf-8"))


def _baseline_summary(payload: dict[str, Any]) -> str:
    if not payload.get("runs"):
        return f"Warning: {payload.get('warning', 'no baseline available')}"
    best = payload.get("best_joint_tradeoff", {})
    return f"Runs: `{len(payload['runs'])}`. Best joint tradeoff: `{best.get('encoder_type')}` with score `{best.get('joint_tradeoff_score')}`."
