"""Test RemoteArtifactManifest schema."""

from optiresearch.schemas.remote_artifacts import (
    RemoteArtifactEntry,
    RemoteArtifactManifest,
)


def test_entry_creation():
    entry = RemoteArtifactEntry(
        artifact_name="result.json",
        relative_path="outputs/result.json",
        artifact_type="result_json",
        required=True,
        sha256="a" * 64,
        size_bytes=1024,
        evidence_role="execution_result",
    )
    assert entry.artifact_name == "result.json"
    assert entry.required is True
    assert entry.evidence_role == "execution_result"


def test_entry_defaults():
    entry = RemoteArtifactEntry(
        artifact_name="test.npz",
        relative_path="outputs/test.npz",
    )
    assert entry.artifact_type == "other"
    assert entry.required is False
    assert entry.evidence_role == "auxiliary"


def test_manifest_creation():
    manifest = RemoteArtifactManifest(
        remote_job_id="job_123",
        run_id="run_456",
        worker_id="windows_wsl",
        artifacts=[
            RemoteArtifactEntry(
                artifact_name="result.json",
                relative_path="result.json",
                artifact_type="result_json",
                required=True,
                evidence_role="execution_result",
            ),
        ],
        completeness="complete",
    )
    assert manifest.schema_version == "0.1"
    assert manifest.completeness == "complete"
    assert len(manifest.artifacts) == 1


def test_manifest_partial_completeness():
    manifest = RemoteArtifactManifest(
        remote_job_id="job_456",
        completeness="partial",
        missing_required_artifacts=["result.json"],
    )
    assert manifest.completeness == "partial"
    assert "result.json" in manifest.missing_required_artifacts
