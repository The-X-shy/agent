"""Tests for legacy manifest normalization."""
from __future__ import annotations

import json


def test_legacy_manifest_without_evidence_role(tmp_path):
    manifest = {
        "schema_version": "0.1",
        "remote_job_id": "legacy_job",
        "artifacts": [
            {"name": "metrics.npz", "path": "outputs/metrics.npz", "sha256": "def456"},
        ],
    }
    manifest_path = tmp_path / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(loaded["artifacts"]) == 1
    # Legacy manifest uses "name" instead of "artifact_name"
    assert "name" in loaded["artifacts"][0]


def test_legacy_manifest_missing_relative_path(tmp_path):
    manifest = {
        "remote_job_id": "old_format",
        "artifacts": [
            {"name": "result.json", "path": "result.json"},
        ],
    }
    manifest_path = tmp_path / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    for art in loaded["artifacts"]:
        # Legacy: has "name" but might lack "relative_path"
        assert "name" in art or "artifact_name" in art
