"""Paper-ready table export for final manuscript.

Exports 10 tables:
  Table 1: System Components
  Table 2: Evidence Levels
  Table 3: Memory Ablation
  Table 4: Optical Encoder Baseline
  Table 5: DeepLens Capability / Realization
  Table 6: HSI Synthetic Baseline
  Table 7: HSI Matrix by Reconstructor
  Table 8: Public Dataset Adapter Status
  Table 9: Claim Whitelist / Blacklist
  Table 10: Limitations and Required Next Evidence
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

from optiresearch.adapters.deeplens import DeepLensAdapter
from optiresearch.hsi.public_datasets import list_hsi_dataset_adapters
from optiresearch.runtime.final_benchmark import FinalBenchmarkRegistry


def export_paper_tables() -> dict[str, Any]:
    root = Path(os.getenv("OPTIRESEARCH_REPORT_ROOT", "./workspace/reports"))
    tables_dir = root / "paper_tables"
    md_dir = tables_dir / "markdown"
    csv_dir = tables_dir / "csv"
    md_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    tables = _build_all_tables()
    _write_markdown_tables(tables, md_dir)
    _write_csv_tables(tables, csv_dir)

    all_md = tables_dir / "all_tables.md"
    all_md.write_text(_all_tables_markdown(tables), encoding="utf-8")

    all_json = tables_dir / "all_tables.json"
    all_json.write_text(json.dumps({"tables": tables}, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "markdown_dir": str(md_dir),
        "csv_dir": str(csv_dir),
        "json_path": str(all_json),
        "all_md": str(all_md),
    }


def _build_all_tables() -> list[dict[str, Any]]:
    registry = FinalBenchmarkRegistry()
    benchmarks = registry.list_benchmarks()
    adapters = list_hsi_dataset_adapters()
    deeplens = DeepLensAdapter().validate_environment()

    return [
        _table_1_system_components(),
        _table_2_evidence_levels(),
        _table_3_memory_ablation(benchmarks),
        _table_4_optical_encoder_baseline(),
        _table_5_deeplens_capability(deeplens),
        _table_6_hsi_synthetic_baseline(),
        _table_7_hsi_matrix_by_reconstructor(),
        _table_8_public_dataset_adapter_status(adapters),
        _table_9_claim_whitelist_blacklist(),
        _table_10_limitations(),
    ]


def _table_1_system_components() -> dict[str, Any]:
    return {
        "id": 1,
        "title": "System Components",
        "headers": ["Component", "Status", "Evidence Level", "Notes"],
        "rows": [
            ["LLM-assisted Agent", "Implemented", "mock", "Mock/deepseek providers available"],
            ["Research Memory OS", "Implemented", "mock", "RunMemory, MetaTrace, SkillMemory"],
            ["Skill Runtime", "Implemented", "mock", "Registry, executor, validator"],
            ["Mock DeepLens Backend", "Implemented", "mock", "Synthetic PSF generation"],
            ["Real DeepLens Adapter", "Conditional", "deeplens_smoke/adapter_proxy/semi_native", "Requires DeepLens SDK"],
            ["Wavelength-aware PSF Contract", "Implemented", "mock", "Interface-level; native physics requires real DeepLens"],
            ["Synthetic HSI Pipeline", "Implemented", "synthetic_hsi", "Forward model + reconstruction"],
            ["Local NPZ Adapter", "Implemented", "public_hsi_mock", "Split/single-file NPZ support"],
            ["CAVE/ICVL Adapters", "Conditional", "public_hsi_mock", "Local-path only; no auto-download"],
            ["Public HSI Matrix", "Implemented", "public_hsi_mock/deeplens_proxy", "Structured skip support"],
            ["ClaimEvidence System", "Implemented", "mock", "Support/contradict/qualify edges"],
            ["DesignRule Manager", "Implemented", "mock", "Rule compilation from claims"],
            ["Paper Protocol v0.1", "Frozen", "N/A", "Experiment protocol frozen"],
        ],
    }


def _table_2_evidence_levels() -> dict[str, Any]:
    return {
        "id": 2,
        "title": "Evidence Levels",
        "headers": ["Evidence Level", "Description", "Availability", "What It Can Support"],
        "rows": [
            ["mock", "Mock/synthetic backend only", "Always available", "System verification, method prototyping"],
            ["deeplens_smoke", "Minimal DeepLens PSF generation", "Requires DeepLens SDK", "Integration-level validation only"],
            ["deeplens_adapter_proxy", "DeepLens via adapter proxy transform", "Requires DeepLens SDK", "Interface contract validation"],
            ["deeplens_semi_native", "DeepLens with semi-native transform", "Requires DeepLens SDK + native features", "Partial physical behavior validation"],
            ["synthetic_hsi", "Synthetic HSI dataset evaluation", "Always available", "Method comparison under controlled conditions"],
            ["public_hsi_mock", "Public/local HSI + mock optical", "Requires local dataset", "Data pipeline validation; NOT real camera"],
            ["public_hsi_deeplens_proxy", "Public HSI + DeepLens proxy", "Requires DeepLens + local dataset", "Optical-HSI integration testing"],
            ["public_hsi_deeplens_semi_native", "Public HSI + semi-native DeepLens", "Requires DeepLens + native features + dataset", "Advanced optical-HSI testing"],
            ["native_optimized", "Native DeepLens optimization", "NOT YET AVAILABLE", "Physical design claims"],
            ["real_lab", "Real laboratory measurements", "NOT YET AVAILABLE", "Real-world performance claims"],
        ],
    }


def _table_3_memory_ablation(benchmarks: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for b in benchmarks:
        if b["group"] == "A_system":
            rows.append([b["name"], b["status"], "See opti-memory benchmark report"])
    return {
        "id": 3,
        "title": "Memory Ablation",
        "headers": ["Ablation", "Status", "Notes"],
        "rows": rows or [
            ["memory_ablation", "available", "OptiMemoryBenchRunner modes: no_memory, trace_only, plan_only, skill_only, full_rmos"],
            ["skill_routing", "available", "SkillRouter with intent matching"],
            ["claim_evidence_rate", "available", "ClaimEvidenceManager review pipeline"],
            ["unsupported_claim_rate", "available", "Claim downgrade rules active"],
            ["llm_fallback_audit", "conditional", "Requires LLM provider"],
        ],
    }


def _table_4_optical_encoder_baseline() -> dict[str, Any]:
    return {
        "id": 4,
        "title": "Optical Encoder Baseline",
        "headers": ["Encoder Type", "Mock Baseline", "DeepLens Smoke", "DeepLens Proxy", "DeepLens Semi-Native"],
        "rows": [
            ["conventional", "Available", "Conditional", "Conditional", "Conditional"],
            ["achromatic", "Available", "Conditional", "Conditional", "Conditional"],
            ["edof", "Available", "Conditional", "Conditional", "Conditional"],
            ["chromatic_coded", "Available", "Conditional", "Conditional", "Conditional"],
            ["controlled_chromatic_edof", "Available", "Conditional", "Conditional", "Conditional"],
        ],
    }


def _table_5_deeplens_capability(deeplens: dict[str, Any]) -> dict[str, Any]:
    caps = deeplens.get("capabilities", [])
    rows = []
    for c in caps:
        rows.append([c.get("name", ""), str(c.get("available", "")), c.get("reason", ""), c.get("evidence", "")])
    return {
        "id": 5,
        "title": "DeepLens Capability / Realization",
        "headers": ["Capability", "Available", "Reason", "Evidence"],
        "rows": rows or [
            ["DeepLens SDK", str(deeplens.get("available", "unknown")), "", ""],
        ],
    }


def _table_6_hsi_synthetic_baseline() -> dict[str, Any]:
    return {
        "id": 6,
        "title": "HSI Synthetic Baseline",
        "headers": ["Metric", "Description", "Status"],
        "rows": [
            ["PSNR", "Peak Signal-to-Noise Ratio", "Computed per run"],
            ["SSIM", "Structural Similarity Index", "Computed per run"],
            ["SAM", "Spectral Angle Mapper", "Computed per run"],
            ["ERGAS", "Relative Global Dimensional Synthesis Error", "Computed per run"],
            ["Reconstruction Score", "Composite ranking score", "Computed per run"],
            ["Coding Strength", "Spectral x (1 - depth) metric", "Computed from PSF features"],
            ["Depth Stability", "PSF similarity across depth planes", "Computed from PSF features"],
            ["Spectral Separability", "PSF variation across wavelength bands", "Computed from PSF features"],
        ],
    }


def _table_7_hsi_matrix_by_reconstructor() -> dict[str, Any]:
    return {
        "id": 7,
        "title": "HSI Matrix by Reconstructor",
        "headers": ["Reconstructor", "Status", "Dependency", "Notes"],
        "rows": [
            ["linear_baseline", "Available", "numpy", "Per-band scalar multipliers"],
            ["optical_conditioned_linear", "Available", "numpy", "Band-dependent spatial basis functions"],
            ["tiny_cnn", "Conditional", "PyTorch", "3-layer CNN; skipped when torch unavailable"],
            ["unet_tiny", "Conditional", "PyTorch", "Tiny UNet; skipped when torch unavailable"],
        ],
    }


def _table_8_public_dataset_adapter_status(adapters: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for dataset_id, item in adapters.items():
        rows.append([dataset_id, str(item.get("available", "")), item.get("download_policy", "")])
    return {
        "id": 8,
        "title": "Public Dataset Adapter Status",
        "headers": ["Dataset", "Available", "Download Policy"],
        "rows": rows or [
            ["synthetic", "True", "Generated on-demand"],
            ["local_npz", "True", "User-provided NPZ files"],
            ["cave", "Conditional", "User-provided local path"],
            ["icvl", "Conditional", "User-provided local path"],
        ],
    }


def _table_9_claim_whitelist_blacklist() -> dict[str, Any]:
    return {
        "id": 9,
        "title": "Claim Whitelist / Blacklist",
        "headers": ["Claim Category", "Status", "Evidence Required", "Example"],
        "rows": [
            ["System executes end-to-end HSI evaluation", "Supported", "mock", "Agent runs full pipeline with metrics output"],
            ["Optical encoder choice affects reconstruction ranking", "Supported", "synthetic_hsi", "Different encoders produce distinct PSNR/SAM"],
            ["Local/public HSI adapter interface works", "Supported", "public_hsi_mock", "NPZ/CAVE/ICVL data preparable through adapter"],
            ["Controlled EDOF > conventional (mock)", "Partially Supported", "synthetic_hsi baseline", "Ranking comparison in mock setting"],
            ["Public HSI + mock optical = pipeline validation", "Qualified", "public_hsi_mock", "NOT real camera validation"],
            ["DeepLens wavelength-aware PSF contract", "Qualified", "deeplens_adapter_proxy", "Interface-level unless native physics available"],
            ["Controlled EDOF best for real HSI", "Unsupported", "native_optimized", "Requires native DeepLens optimization"],
            ["DeepLens proxy = native physical validation", "Unsupported", "deeplens_native", "Proxy results != native physics"],
            ["Public dataset + mock = real optical design proof", "Unsupported", "real_lab", "Mock measurement != real camera"],
        ],
    }


def _table_10_limitations() -> dict[str, Any]:
    return {
        "id": 10,
        "title": "Limitations and Required Next Evidence",
        "headers": ["Limitation", "Current Evidence Level", "Required Next Evidence", "Priority"],
        "rows": [
            ["No real camera HSI performance", "synthetic_hsi / public_hsi_mock", "Real lab measurements", "High"],
            ["No native DeepLens validation", "deeplens_adapter_proxy", "Native DeepLens optimization", "High"],
            ["No native optimized EDOF-HSI claim", "synthetic_hsi", "Native optimization loop", "High"],
            ["Tiny CNN / UNet require PyTorch", "Conditional", "Optional dependency", "Low"],
            ["CAVE/ICVL require local data", "Conditional", "User-provided dataset paths", "Low"],
            ["LLM claims require provider", "Conditional", "LLM provider configuration", "Medium"],
        ],
    }


def _write_markdown_tables(tables: list[dict[str, Any]], md_dir: Path) -> None:
    for t in tables:
        name = t["title"].lower().replace(" ", "_").replace("/", "_")
        path = md_dir / f"table_{t['id']}_{name}.md"
        lines = [f"# Table {t['id']}: {t['title']}", ""]
        lines.append("| " + " | ".join(t["headers"]) + " |")
        lines.append("|" + "|".join(["---" for _ in t["headers"]]) + "|")
        for row in t["rows"]:
            lines.append("| " + " | ".join(str(c) for c in row) + " |")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv_tables(tables: list[dict[str, Any]], csv_dir: Path) -> None:
    for t in tables:
        name = t["title"].lower().replace(" ", "_").replace("/", "_")
        path = csv_dir / f"table_{t['id']}_{name}.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(t["headers"])
            for row in t["rows"]:
                writer.writerow(row)


def _all_tables_markdown(tables: list[dict[str, Any]]) -> str:
    lines = ["# Paper-Ready Tables", ""]
    for t in tables:
        lines.append(f"## Table {t['id']}: {t['title']}")
        lines.append("")
        lines.append("| " + " | ".join(t["headers"]) + " |")
        lines.append("|" + "|".join(["---" for _ in t["headers"]]) + "|")
        for row in t["rows"]:
            lines.append("| " + " | ".join(str(c) for c in row) + " |")
        lines.append("")
    return "\n".join(lines)
