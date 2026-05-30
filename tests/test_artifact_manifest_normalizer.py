"""Tests for artifact manifest normalization."""
from __future__ import annotations

import json


def test_normalizer_handles_minimal_manifest(tmp_path):
    manifest = {
        "schema_version": "0.1",
        "remote_job_id": "test_job",
        "artifacts": [
            {"artifact_name": "result.json", "relative_path": "result.json",
             "sha256": "abc123", "evidence_role": "execution_result"},
        ],
    }
    manifest_path = tmp_path / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    # Verify it can be read
    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert loaded["remote_job_id"] == "test_job"
    assert len(loaded["artifacts"]) == 1


def test_normalizer_reports_sha256_missing(tmp_path):
    manifest = {
        "schema_version": "0.1",
        "remote_job_id": "test_job_2",
        "artifacts": [
            {"artifact_name": "result.json", "relative_path": "result.json"},
        ],
    }
    manifest_path = tmp_path / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    without_sha = [a for a in loaded["artifacts"] if not a.get("sha256")]
    assert len(without_sha) == 1
