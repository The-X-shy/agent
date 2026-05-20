"""Final paper package exporter.

Assembles all paper-ready artifacts into a single distributable package:
  workspace/final_paper_package/
  ├── README.md
  ├── final_benchmark_summary.md
  ├── paper_tables/
  ├── claim_boundary.md
  ├── evidence_distribution.md
  ├── paper_experiment_protocol_v0.1_freeze.md
  ├── phase_reports/
  ├── artifact_inventory.json
  └── reproducibility_manifest.json
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from optiresearch.adapters.deeplens import DeepLensAdapter
from optiresearch.hsi.public_datasets import list_hsi_dataset_adapters
from optiresearch.reports.claim_boundary import generate_claim_whitelist_blacklist
from optiresearch.reports.evidence_distribution import compute_evidence_distribution
from optiresearch.reports.paper_tables import export_paper_tables
from optiresearch.runtime.final_benchmark import FinalBenchmarkRegistry


def export_final_paper_package(output_dir: Path | None = None) -> dict[str, Any]:
    root = output_dir or Path("./workspace/final_paper_package")
    root.mkdir(parents=True, exist_ok=True)

    registry = FinalBenchmarkRegistry()
    claim_boundary = generate_claim_whitelist_blacklist()
    evidence_dist = compute_evidence_distribution()
    paper_tables_result = export_paper_tables()

    # Write benchmark summary (export_summary writes directly into root)
    registry.export_summary(root)

    # Write paper tables
    tables_dest = root / "paper_tables"
    tables_dest.mkdir(parents=True, exist_ok=True)
    tables_src = Path(paper_tables_result["markdown_dir"])
    if tables_src.exists():
        for f in tables_src.iterdir():
            shutil.copy2(f, tables_dest / f.name)
    all_tables_src = Path(paper_tables_result["all_md"])
    if all_tables_src.exists():
        shutil.copy2(all_tables_src, tables_dest / "all_tables.md")

    # Write claim boundary
    _write_boundary_md(claim_boundary, root / "claim_boundary.md")
    (root / "claim_boundary.json").write_text(
        json.dumps(claim_boundary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Write evidence distribution
    _write_evidence_md(evidence_dist, root / "evidence_distribution.md")
    (root / "evidence_distribution.json").write_text(
        json.dumps(evidence_dist, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Write artifact inventory
    inventory = {
        "required_artifacts": registry.validate_required_artifacts(),
        "benchmark_items": [
            {"group": b["group"], "name": b["name"], "status": b["status"]}
            for b in registry.list_benchmarks()
        ],
    }
    (root / "artifact_inventory.json").write_text(
        json.dumps(inventory, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Copy phase reports
    phase_dir = root / "phase_reports"
    phase_dir.mkdir(parents=True, exist_ok=True)
    reports_root = Path(os.getenv("OPTIRESEARCH_REPORT_ROOT", "./workspace/reports"))
    for name in [
        "phase10_optical_sensitive_hsi_report.md",
        "phase11_hsi_network_dataset_report.md",
        "phase12_public_hsi_deeplens_protocol_report.md",
        "paper_experiment_protocol_v0.1_freeze.md",
    ]:
        src = reports_root / name
        if src.exists():
            shutil.copy2(src, phase_dir / name)

    # Write reproducibility manifest
    manifest = _build_reproducibility_manifest()
    manifest_path = root / "reproducibility_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    # Write README
    (root / "README.md").write_text(_package_readme(manifest), encoding="utf-8")

    return {
        "package_dir": str(root),
        "manifest_path": str(manifest_path),
    }


def _build_reproducibility_manifest() -> dict[str, Any]:
    deeplens = DeepLensAdapter().validate_environment()
    adapters = list_hsi_dataset_adapters()

    code_version = "0.1.0"
    try:
        import subprocess
        result = subprocess.run(["git", "describe", "--always", "--dirty"], capture_output=True, text=True)
        if result.returncode == 0:
            code_version = result.stdout.strip()
    except Exception:
        pass

    return {
        "code_version": code_version,
        "python_version": sys.version,
        "platform": platform.platform(),
        "package_summary": "OptiResearch Agent v0.1.0 — computational optics research automation",
        "workspace_paths": {
            "report_root": os.getenv("OPTIRESEARCH_REPORT_ROOT", "./workspace/reports"),
            "hsi_root": os.getenv("OPTIRESEARCH_HSI_ROOT", "./workspace/hsi"),
            "benchmark_root": os.getenv("OPTIRESEARCH_BENCHMARK_ROOT", "./workspace/benchmarks"),
            "db_path": os.getenv("OPTIRESEARCH_DB_PATH", "./workspace/optiresearch.sqlite"),
        },
        "dataset_availability": {
            dataset_id: {"available": item.get("available"), "download_policy": item.get("download_policy")}
            for dataset_id, item in adapters.items()
        },
        "deeplens_availability": {
            "available": deeplens.get("available"),
            "version": deeplens.get("deeplens_version"),
        },
        "llm_provider_availability": "conditional — requires provider configuration",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "limitations": [
            "No real camera HSI performance data.",
            "No native DeepLens wavelength physics validation.",
            "No native optimized EDOF-HSI claim.",
            "Synthetic/mock results must not be presented as real HSI performance.",
            "DeepLens adapter_proxy must not be presented as native validation.",
            "Public dataset + mock optical must not be presented as real camera experiment.",
        ],
    }


def _write_boundary_md(boundary: dict[str, Any], path: Path) -> None:
    lines = ["# Claim Boundary", ""]
    for cat, title in [
        ("supported_claims", "Supported Claims"),
        ("qualified_claims", "Qualified Claims"),
        ("unsupported_claims", "Unsupported Claims"),
    ]:
        lines.append(f"## {title}")
        lines.append("")
        for c in boundary.get(cat, []):
            lines.append(f"- **{c['text']}** — {c['rationale']}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_evidence_md(dist: dict[str, Any], path: Path) -> None:
    lines = [
        "# Evidence Distribution",
        "",
        "## Count by Evidence Level",
        "",
        "| Evidence Level | Count |",
        "|---|---|",
    ]
    for level, count in dist.get("count_by_level", {}).items():
        lines.append(f"| {level} | {count} |")
    lines.extend([
        "",
        "## Status Distribution",
        "",
        "| Status | Count |",
        "|---|---|",
    ])
    for status, count in dist.get("status_counts", {}).items():
        lines.append(f"| {status} | {count} |")
    if dist.get("missing_evidence_warnings"):
        lines.extend(["", "## Missing Evidence Warnings", ""])
        for w in dist["missing_evidence_warnings"]:
            lines.append(f"- {w}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _copy_file(src: Path, dest: Path) -> None:
    if src.exists():
        shutil.copy2(src, dest)


def _package_readme(manifest: dict[str, Any]) -> str:
    lines = [
        "# Final Paper Package — OptiResearch Agent",
        "",
        f"**Generated:** {manifest['timestamp']}",
        f"**Code version:** {manifest['code_version']}",
        f"**Python:** {manifest['python_version'].split()[0]}",
        "",
        "## Contents",
        "",
        "- `final_benchmark_summary.md` — 5-group benchmark registry summary",
        "- `paper_tables/` — 10 paper-ready tables (Markdown + CSV + JSON)",
        "- `claim_boundary.md` — Supported / Qualified / Unsupported claims",
        "- `evidence_distribution.md` — Evidence level and status distribution",
        "- `artifact_inventory.json` — Complete artifact inventory",
        "- `reproducibility_manifest.json` — Environment and availability manifest",
        "- `phase_reports/` — Phase 10-12 reports + frozen protocol",
        "",
        "## Reproducibility",
        "",
        "This package captures the complete paper-ready evidence bundle for the OptiResearch Agent.",
        "All claims are bounded by evidence levels documented in `claim_boundary.md`.",
        "",
        "## Limitations",
        "",
    ]
    for limit in manifest.get("limitations", []):
        lines.append(f"- {limit}")
    lines.extend([
        "",
        "## Usage",
        "",
        "```bash",
        "python -m optiresearch.cli export-final-paper-package",
        "```",
    ])
    return "\n".join(lines)
