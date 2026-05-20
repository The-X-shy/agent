"""Evidence level distribution statistics.

Computes counts by evidence level and claim status, artifact coverage,
and missing evidence warnings from the ClaimEvidenceManager database.
"""

from __future__ import annotations

from typing import Any

from optiresearch.memory.claim_evidence import ClaimEvidenceManager


def compute_evidence_distribution() -> dict[str, Any]:
    manager = ClaimEvidenceManager()
    claims = manager.list_claims()

    count_by_level: dict[str, int] = {
        "mock": 0,
        "deeplens_smoke": 0,
        "deeplens_adapter_proxy": 0,
        "deeplens_semi_native": 0,
        "synthetic_hsi": 0,
        "public_hsi_mock": 0,
        "public_hsi_deeplens_proxy": 0,
        "public_hsi_deeplens_semi_native": 0,
        "native_optimized": 0,
        "real_lab": 0,
    }

    status_counts: dict[str, int] = {
        "supported": 0,
        "partially_supported": 0,
        "unsupported": 0,
        "contradicted": 0,
        "needs_followup": 0,
        "simulation_only": 0,
        "prototype_validated": 0,
    }

    artifact_coverage = {"claims_with_artifacts": 0, "claims_without_artifacts": 0}
    missing_warnings: list[str] = []

    for claim in claims:
        level = claim.metadata.get("evidence_level") or claim.metadata.get("optical_backend_evidence_level", "mock")
        if level in count_by_level:
            count_by_level[level] += 1
        else:
            count_by_level["mock"] += 1

        status = claim.status
        if status in status_counts:
            status_counts[status] += 1

        if claim.support_edges:
            artifact_coverage["claims_with_artifacts"] += 1
        else:
            artifact_coverage["claims_without_artifacts"] += 1

        if claim.warnings:
            for w in claim.warnings:
                if "missing" in w.lower():
                    missing_warnings.append(f"{claim.claim_id}: {w}")

    if count_by_level["native_optimized"] == 0:
        missing_warnings.append("No native_optimized evidence level claims found — native DeepLens optimization not yet performed.")
    if count_by_level["real_lab"] == 0:
        missing_warnings.append("No real_lab evidence level claims found — real laboratory validation not yet performed.")

    return {
        "count_by_level": count_by_level,
        "status_counts": status_counts,
        "artifact_coverage": artifact_coverage,
        "missing_evidence_warnings": missing_warnings,
    }
