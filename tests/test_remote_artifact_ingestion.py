"""Test remote artifact ingestion."""

import json
import tempfile
from pathlib import Path

from optiresearch.remote.artifact_ingestion import (
    ingest_remote_artifact_manifest,
    validate_remote_artifact_manifest,
)


def test_validate_valid_manifest():
    manifest = {
        "schema_version": "0.1",
        "remote_job_id": "job_test",
        "run_id": "run_test",
        "worker_id": "windows_wsl",
        "completeness": "complete",
        "missing_required_artifacts": [],
        "artifacts": [
            {
                "artifact_name": "result.json",
                "relative_path": "result.json",
                "artifact_type": "result_json",
                "required": True,
                "evidence_role": "execution_result",
            },
        ],
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(manifest, f)
        path = f.name

    try:
        result = validate_remote_artifact_manifest(path)
        assert result["valid"] is True
        assert result["artifact_count"] == 1
    finally:
        Path(path).unlink(missing_ok=True)


def test_validate_missing_file():
    result = validate_remote_artifact_manifest("/nonexistent/manifest.json")
    assert result["valid"] is False


def test_ingest_missing_manifest():
    result = ingest_remote_artifact_manifest("/nonexistent/manifest.json")
    assert result.errors


def test_validate_incomplete_manifest():
    manifest = {
        "schema_version": "0.1",
        "remote_job_id": "job_test",
        "completeness": "complete",
        "missing_required_artifacts": ["result.json"],
        "artifacts": [],
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(manifest, f)
        path = f.name

    try:
        result = validate_remote_artifact_manifest(path)
        assert result["valid"] is False  # completeness mismatch
    finally:
        Path(path).unlink(missing_ok=True)
