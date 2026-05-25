"""Fixture-based artifact-bound claim evidence success path test."""

import json
import tempfile
from pathlib import Path

from optiresearch.memory.claim_evidence import (
    bind_artifacts_from_claim_gate_decision,
    ClaimEvidenceManager,
)
from optiresearch.storage.file_artifact_store import FileArtifactStore


def test_fixture_artifact_bound_success_path():
    """Create fixture artifacts, register in store, bind to ClaimEvidence."""
    store = FileArtifactStore()
    workspace_id = "fixture_test"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        result_path = tmp / "result.json"
        metrics_path = tmp / "metrics.json"
        result_path.write_text(json.dumps({"status": "succeeded"}))
        metrics_path.write_text(json.dumps({
            "reconstruction_loss_before": 0.1,
            "reconstruction_loss_after": 0.05,
            "improvement_detected": True,
        }))

        result_ref = store.register_file(
            str(result_path), workspace_id=workspace_id,
            run_id="fixture_run_001", trace_id="fixture_trace_001",
            producer="fixture",
            metadata={"artifact_type": "result_json", "evidence_role": "execution_result",
                       "remote_job_id": "fixture_job_001"},
            metrics={"reconstruction_loss_after": 0.05},
        )
        result_aid = result_ref.artifact_id
        metrics_ref = store.register_file(
            str(metrics_path), workspace_id=workspace_id,
            run_id="fixture_run_001", trace_id="fixture_trace_002",
            producer="fixture",
            metadata={"artifact_type": "metrics_json", "evidence_role": "primary_metric",
                       "remote_job_id": "fixture_job_001"},
            metrics={"reconstruction_loss_after": 0.05, "improvement_detected": True},
        )
        metrics_aid = metrics_ref.artifact_id

        decision = {
            "evidence_artifact_ids": [result_aid, metrics_aid],
            "primary_metric_artifact_id": metrics_aid,
            "execution_result_artifact_id": result_aid,
            "evidence_completeness": "complete",
            "missing_evidence_artifacts": [],
        }

        edges = bind_artifacts_from_claim_gate_decision(
            "fixture_claim_001", decision, artifact_store=store,
        )
        assert len(edges) >= 2
        roles = {e["evidence_role"] for e in edges}
        assert "primary_metric" in roles
        assert "execution_result" in roles


def test_fixture_binding_creates_claim_evidence():
    """Full flow: register → bind → create claim → attach edges."""
    manager = ClaimEvidenceManager(workspace_id="fixture_test")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        r = tmp / "r.json"
        r.write_text(json.dumps({"status": "succeeded"}))
        artifact_store = FileArtifactStore()
        ref = artifact_store.register_file(
            str(r), workspace_id="fixture_test",
            run_id="fr2", trace_id="ft2",
            producer="fixture",
            metadata={"evidence_role": "execution_result", "remote_job_id": "fj2"},
            metrics={},
        )
        aid = ref.artifact_id

    decision = {
        "evidence_artifact_ids": [aid],
        "primary_metric_artifact_id": aid,
        "execution_result_artifact_id": "",
        "evidence_completeness": "complete",
        "missing_evidence_artifacts": [],
    }

    claim = manager.create_claim("Fixture bound claim", {"execution_id": "fixture_exec"})
    edges = bind_artifacts_from_claim_gate_decision(claim.claim_id, decision,
                                                     artifact_store=artifact_store)
    for e in edges:
        manager.attach_support(claim.claim_id, e["artifact_id"], e["score"])

    retrieved = manager.get_claim(claim.claim_id)
    assert retrieved is not None
    assert len(retrieved.support_edges) >= 1


def test_empty_artifacts_returns_no_edges():
    decision = {
        "evidence_artifact_ids": [],
        "primary_metric_artifact_id": "",
        "execution_result_artifact_id": "",
        "evidence_completeness": "partial",
        "missing_evidence_artifacts": ["result.json"],
    }
    edges = bind_artifacts_from_claim_gate_decision("empty_claim", decision)
    assert len(edges) == 0
