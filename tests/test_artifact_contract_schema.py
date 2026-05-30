"""Tests for ArtifactContract schema."""
from __future__ import annotations

import pytest

from optiresearch.schemas.artifact_contract import ArtifactContract


def test_artifact_contract_minimal():
    contract = ArtifactContract(contract_id="ac_diagnostic")
    assert contract.contract_id == "ac_diagnostic"
    assert contract.sha256_required is True
    assert contract.missing_artifact_policy == "structured_warning"


def test_artifact_contract_full():
    contract = ArtifactContract(
        contract_id="ac_native_geolens",
        handler_id="deeplens_native_geolens_hsi_codesign",
        required_artifacts=["result.json", "metrics.json", "artifact_manifest.json", "report.md"],
        optional_artifacts=["psf_stats.json", "stability_trace.json", "benchmark_results.csv"],
        artifact_roles={
            "result.json": "execution_result",
            "metrics.json": "primary_metric",
            "report.md": "report",
            "psf_stats.json": "psf_artifact",
        },
        sha256_required=True,
        artifactstore_registration_required=True,
        evidence_binding_required=True,
        missing_artifact_policy="needs_followup",
    )
    assert len(contract.required_artifacts) == 4
    assert contract.artifact_roles["result.json"] == "execution_result"
    assert contract.evidence_binding_required is True


def test_artifact_contract_rejects_extra():
    with pytest.raises(ValueError):
        ArtifactContract(contract_id="test", bad_field=1)
