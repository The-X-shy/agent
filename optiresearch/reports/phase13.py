"""Phase 13 final benchmark report."""

from __future__ import annotations

import os
from pathlib import Path

from optiresearch.adapters.deeplens import DeepLensAdapter
from optiresearch.hsi.public_datasets import list_hsi_dataset_adapters
from optiresearch.reports.claim_boundary import generate_claim_whitelist_blacklist
from optiresearch.reports.evidence_distribution import compute_evidence_distribution
from optiresearch.runtime.final_benchmark import FinalBenchmarkRegistry


def export_phase13_report() -> Path:
    root = Path(os.getenv("OPTIRESEARCH_REPORT_ROOT", "./workspace/reports"))
    root.mkdir(parents=True, exist_ok=True)
    path = root / "phase13_final_benchmark_report.md"
    path.write_text(_markdown(), encoding="utf-8")
    return path


def _markdown() -> str:
    registry = FinalBenchmarkRegistry()
    benchmarks = registry.list_benchmarks()
    validation = registry.validate_required_artifacts()
    claim_boundary = generate_claim_whitelist_blacklist()
    evidence_dist = compute_evidence_distribution()
    deeplens = DeepLensAdapter().validate_environment()
    adapters = list_hsi_dataset_adapters()

    lines = [
        "# Phase 13: Final Benchmark Freeze and Paper Evidence Package",
        "",
        "## 1. Objective",
        "",
        "Freeze the paper-ready benchmark structure, generate all paper-ready tables, establish claim boundaries, compute evidence distributions, and produce the final reproducibility package.",
        "",
        "## 2. System Maturity Summary",
        "",
        "The OptiResearch Agent has completed 13 development phases. The system now includes:",
        "",
        "- LLM-assisted Agent with mock and DeepSeek providers.",
        "- Research Memory OS (RunMemory, MetaTrace, SkillMemory).",
        "- Skill Runtime (registry, executor, validator).",
        "- Mock / DeepLens optical backend with 5 encoder types.",
        "- DeepLens wavelength-aware PSF contract.",
        "- Synthetic / local_npz / CAVE / ICVL dataset adapters.",
        "- Public HSI matrix with structured skip support.",
        "- ClaimEvidence pipeline with downgrade rules.",
        "- DesignRule memory compilation.",
        "- Frozen paper experiment protocol v0.1.",
        "",
        "## 3. Final Benchmark Registry",
        "",
        f"**Total benchmarks:** {len(benchmarks)}",
        f"**Validation status:** {validation['status']}",
        "",
        "### Group A: System Benchmark",
        "",
        "| Name | Status |",
        "|---|---|",
    ]
    for b in benchmarks:
        if b["group"] == "A_system":
            lines.append(f"| {b['name']} | {b['status']} |")

    lines.extend([
        "",
        "### Group B: Optical Backend Benchmark",
        "",
        "| Name | Status |",
        "|---|---|",
    ])
    for b in benchmarks:
        if b["group"] == "B_optical_backend":
            lines.append(f"| {b['name']} | {b['status']} |")

    lines.extend([
        "",
        "### Group C: HSI Synthetic Benchmark",
        "",
        "| Name | Status |",
        "|---|---|",
    ])
    for b in benchmarks:
        if b["group"] == "C_hsi_synthetic":
            lines.append(f"| {b['name']} | {b['status']} |")

    lines.extend([
        "",
        "### Group D: Public/Local HSI Benchmark",
        "",
        "| Name | Status |",
        "|---|---|",
    ])
    for b in benchmarks:
        if b["group"] == "D_public_local_hsi":
            lines.append(f"| {b['name']} | {b['status']} |")

    lines.extend([
        "",
        "### Group E: Evidence Benchmark",
        "",
        "| Name | Status |",
        "|---|---|",
    ])
    for b in benchmarks:
        if b["group"] == "E_evidence":
            lines.append(f"| {b['name']} | {b['status']} |")

    lines.extend([
        "",
        "## 4. Paper-ready Tables",
        "",
        "10 tables exported to `workspace/reports/paper_tables/`:",
        "",
        "1. System Components",
        "2. Evidence Levels",
        "3. Memory Ablation",
        "4. Optical Encoder Baseline",
        "5. DeepLens Capability / Realization",
        "6. HSI Synthetic Baseline",
        "7. HSI Matrix by Reconstructor",
        "8. Public Dataset Adapter Status",
        "9. Claim Whitelist / Blacklist",
        "10. Limitations and Required Next Evidence",
        "",
        "## 5. Claim Boundary",
        "",
        f"**Supported claims:** {len(claim_boundary['supported_claims'])}",
        f"**Qualified claims:** {len(claim_boundary['qualified_claims'])}",
        f"**Unsupported claims:** {len(claim_boundary['unsupported_claims'])}",
        "",
        "### Supported Claims",
        "",
    ])
    for c in claim_boundary["supported_claims"]:
        lines.append(f"- {c['text']}")

    lines.extend([
        "",
        "### Qualified Claims",
        "",
    ])
    for c in claim_boundary["qualified_claims"]:
        lines.append(f"- {c['text']}")

    lines.extend([
        "",
        "### Unsupported Claims",
        "",
    ])
    for c in claim_boundary["unsupported_claims"]:
        lines.append(f"- {c['text']}")

    lines.extend([
        "",
        "## 6. Evidence Distribution",
        "",
        "| Evidence Level | Count |",
        "|---|---|",
    ])
    for level, count in evidence_dist["count_by_level"].items():
        lines.append(f"| {level} | {count} |")

    lines.extend([
        "",
        "| Status | Count |",
        "|---|---|",
    ])
    for status, count in evidence_dist["status_counts"].items():
        lines.append(f"| {status} | {count} |")

    lines.extend([
        "",
        "## 7. Dataset and Backend Readiness",
        "",
        f"DeepLens available: `{deeplens.get('available')}`",
        "",
        "| Dataset | Available | Download Policy |",
        "|---|---|---|",
    ])
    for dataset_id, item in adapters.items():
        lines.append(f"| {dataset_id} | {item['available']} | {item['download_policy']} |")

    lines.extend([
        "",
        "## 8. Reproducibility Package",
        "",
        "The final paper package is exported via:",
        "",
        "```bash",
        "python -m optiresearch.cli export-final-paper-package",
        "```",
        "",
        "Output: `workspace/final_paper_package/`",
        "",
        "## 9. What Is Ready for Paper Writing",
        "",
        "- System architecture description (all components implemented).",
        "- Evidence level framework and claim boundary documentation.",
        "- Mock/synthetic HSI baseline results with encoder ranking.",
        "- Public dataset adapter interface specification.",
        "- DeepLens wavelength-aware PSF contract specification.",
        "- Memory ablation benchmark methodology.",
        "- Paper-ready tables (10 tables in MD/CSV/JSON).",
        "- Frozen experiment protocol v0.1.",
        "",
        "## 10. What Still Requires Native Optimization / Real Lab Evidence",
        "",
        "- Real camera HSI performance claims (need lab measurements).",
        "- Native DeepLens wavelength physics validation (need full SDK).",
        "- Native optimized EDOF-HSI claim (need native optimization loop).",
        "- Comparison with published real-HSI methods (need real data).",
        "- Physical optical design validation (need real measurements).",
        "",
        "## 11. Phase 14 Recommendation",
        "",
        "Phase 14 should focus on:",
        "",
        "1. **Native DeepLens optimization**: Full SDK integration for native encoder optimization.",
        "2. **Real lab validation**: Laboratory HSI measurements with physical camera setup.",
        "3. **Final manuscript drafting**: Use the paper writing assistant agent with the frozen evidence package.",
        "4. **External benchmark comparison**: Compare against published computational imaging methods.",
        "",
        "The current evidence package provides a complete system description and mock/synthetic validation.",
        "Real-world performance claims require Phase 14 native optimization and lab validation.",
    ])
    return "\n".join(lines)
