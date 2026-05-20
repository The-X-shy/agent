"""Final paper-ready benchmark registry.

Freezes the benchmark structure into 5 groups:
  A. System benchmark
  B. Optical backend benchmark
  C. HSI synthetic benchmark
  D. Public/local HSI benchmark
  E. Evidence benchmark
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class FinalBenchmarkRegistry:
    """List, validate, collect, and export the 5-group final benchmark."""

    def list_benchmarks(self) -> list[dict[str, Any]]:
        return [
            {"group": "A_system", "name": "memory_ablation", "status": "available"},
            {"group": "A_system", "name": "skill_routing", "status": "available"},
            {"group": "A_system", "name": "claim_evidence_rate", "status": "available"},
            {"group": "A_system", "name": "unsupported_claim_rate", "status": "available"},
            {"group": "A_system", "name": "llm_fallback_audit", "status": "conditional"},
            {"group": "B_optical_backend", "name": "mock_encoder_baseline", "status": "available"},
            {"group": "B_optical_backend", "name": "deeplens_smoke", "status": "conditional"},
            {"group": "B_optical_backend", "name": "deeplens_adapter_proxy", "status": "conditional"},
            {"group": "B_optical_backend", "name": "deeplens_semi_native", "status": "conditional"},
            {"group": "B_optical_backend", "name": "wavelength_aware_psf_contract", "status": "available"},
            {"group": "C_hsi_synthetic", "name": "optical_sensitive_hsi_baseline", "status": "available"},
            {"group": "C_hsi_synthetic", "name": "reconstructor_matrix", "status": "available"},
            {"group": "C_hsi_synthetic", "name": "encoder_ranking_by_reconstructor", "status": "available"},
            {"group": "D_public_local_hsi", "name": "local_npz_adapter", "status": "available"},
            {"group": "D_public_local_hsi", "name": "cave_adapter", "status": "conditional"},
            {"group": "D_public_local_hsi", "name": "icvl_adapter", "status": "conditional"},
            {"group": "D_public_local_hsi", "name": "public_hsi_matrix", "status": "available"},
            {"group": "D_public_local_hsi", "name": "structured_skip", "status": "available"},
            {"group": "E_evidence", "name": "claim_whitelist", "status": "available"},
            {"group": "E_evidence", "name": "claim_blacklist", "status": "available"},
            {"group": "E_evidence", "name": "design_rule_status", "status": "available"},
            {"group": "E_evidence", "name": "evidence_level_distribution", "status": "available"},
        ]

    def validate_required_artifacts(self) -> dict[str, Any]:
        present: list[str] = []
        missing: list[str] = []
        required = [
            "workspace/reports/paper_experiment_protocol_v0.1_freeze.md",
            "workspace/reports/phase10_optical_sensitive_hsi_report.md",
            "workspace/reports/phase11_hsi_network_dataset_report.md",
            "workspace/reports/phase12_public_hsi_deeplens_protocol_report.md",
        ]
        for rel in required:
            if Path(rel).exists():
                present.append(rel)
            else:
                missing.append(rel)
        status = "ready" if not missing else "partial"
        return {"status": status, "present": present, "missing": missing}

    def collect_results(self) -> dict[str, Any]:
        return {
            "A_system": self._collect_system(),
            "B_optical_backend": self._collect_optical_backend(),
            "C_hsi_synthetic": self._collect_hsi_synthetic(),
            "D_public_local_hsi": self._collect_public_local_hsi(),
            "E_evidence": self._collect_evidence(),
        }

    def export_summary(self, output_dir: Path) -> dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        results = self.collect_results()
        validation = self.validate_required_artifacts()
        benchmarks = self.list_benchmarks()

        summary = {
            "benchmark_count": len(benchmarks),
            "groups": list(results.keys()),
            "validation": validation,
            "results": results,
        }

        summary_json = output_dir / "final_benchmark_summary.json"
        summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

        summary_md = output_dir / "final_benchmark_summary.md"
        summary_md.write_text(self._summary_markdown(summary, benchmarks), encoding="utf-8")

        inventory = output_dir / "artifact_inventory.json"
        inventory.write_text(json.dumps({
            "required_artifacts": validation,
            "benchmark_items": [{"group": b["group"], "name": b["name"], "status": b["status"]} for b in benchmarks],
        }, indent=2, ensure_ascii=False), encoding="utf-8")

        return {
            "summary_json": str(summary_json),
            "summary_md": str(summary_md),
            "artifact_inventory": str(inventory),
        }

    def _collect_system(self) -> dict[str, Any]:
        return {
            "memory_ablation": {"status": "available", "runner": "OptiMemoryBenchRunner"},
            "skill_routing": {"status": "available"},
            "claim_evidence_rate": {"status": "available"},
            "unsupported_claim_rate": {"status": "available"},
            "llm_fallback_audit": {"status": "conditional", "requires": "LLM provider"},
        }

    def _collect_optical_backend(self) -> dict[str, Any]:
        return {
            "mock_encoder_baseline": {"status": "available"},
            "deeplens_smoke": {"status": "conditional", "requires": "OPTIRESEARCH_ENABLE_REAL_DEEPLENS_TESTS=1"},
            "deeplens_adapter_proxy": {"status": "conditional", "requires": "OPTIRESEARCH_ENABLE_REAL_DEEPLENS_TESTS=1"},
            "deeplens_semi_native": {"status": "conditional", "requires": "OPTIRESEARCH_ENABLE_REAL_DEEPLENS_TESTS=1"},
            "wavelength_aware_psf_contract": {"status": "available"},
        }

    def _collect_hsi_synthetic(self) -> dict[str, Any]:
        return {
            "optical_sensitive_hsi_baseline": {"status": "available"},
            "reconstructor_matrix": {"status": "available"},
            "encoder_ranking_by_reconstructor": {"status": "available"},
        }

    def _collect_public_local_hsi(self) -> dict[str, Any]:
        return {
            "local_npz_adapter": {"status": "available"},
            "cave_adapter": {"status": "conditional", "requires": "local CAVE dataset path"},
            "icvl_adapter": {"status": "conditional", "requires": "local ICVL dataset path"},
            "public_hsi_matrix": {"status": "available"},
            "structured_skip": {"status": "available"},
        }

    def _collect_evidence(self) -> dict[str, Any]:
        return {
            "claim_whitelist": {"status": "available"},
            "claim_blacklist": {"status": "available"},
            "design_rule_status": {"status": "available"},
            "evidence_level_distribution": {"status": "available"},
        }

    def _summary_markdown(self, summary: dict[str, Any], benchmarks: list[dict[str, Any]]) -> str:
        lines = [
            "# Final Benchmark Summary",
            "",
            f"**Total benchmarks:** {summary['benchmark_count']}",
            f"**Validation status:** {summary['validation']['status']}",
            "",
            "## Benchmark Groups",
            "",
        ]
        for group in ["A_system", "B_optical_backend", "C_hsi_synthetic", "D_public_local_hsi", "E_evidence"]:
            lines.append(f"### Group {group}")
            lines.append("")
            lines.append("| Name | Status |")
            lines.append("|---|---|")
            for b in benchmarks:
                if b["group"] == group:
                    lines.append(f"| {b['name']} | {b['status']} |")
            lines.append("")
        if summary["validation"]["missing"]:
            lines.append("## Missing Artifacts")
            lines.append("")
            for m in summary["validation"]["missing"]:
                lines.append(f"- {m}")
            lines.append("")
        return "\n".join(lines)
