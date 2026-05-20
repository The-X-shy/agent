import json

from optiresearch.remote.result_ingestion import ingest_remote_job_result
from optiresearch.schemas.remote import RemoteJobResult
from optiresearch.storage.sqlite_store import SQLiteStore


def test_ingestion_registers_artifacts_memory_and_claim(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_DB_PATH", str(tmp_path / "memory.sqlite"))
    monkeypatch.setenv("OPTIRESEARCH_ARTIFACT_ROOT", str(tmp_path / "artifacts"))

    local_dir = tmp_path / "remote_jobs" / "job_1"
    local_dir.mkdir(parents=True)
    manifest_path = local_dir / "source_smoke_manifest.json"
    manifest_path.write_text(json.dumps({"available": True, "import_path": "/x/deeplens/__init__.py"}), encoding="utf-8")
    metrics_path = local_dir / "metrics_summary.json"
    metrics_path.write_text(json.dumps({"available": True, "fallback_used": False}), encoding="utf-8")
    (local_dir / "artifact_manifest.json").write_text(
        json.dumps(
            {
                "artifacts": [
                    {
                        "path": "source_smoke_manifest.json",
                        "artifact_type": "manifest",
                        "metrics": {"available": 1},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    result = RemoteJobResult(
        job_id="job_1",
        status="succeeded",
        remote_run_id="remote_smoke",
        started_at="2026-05-20T00:00:00Z",
        finished_at="2026-05-20T00:00:01Z",
        command=["python", "-m", "optiresearch.cli", "run-deeplens-source-smoke"],
        stdout_path=str(local_dir / "stdout.txt"),
        stderr_path=str(local_dir / "stderr.txt"),
        remote_output_dir="/mnt/d/agent/workspace/remote_jobs/job_1",
        local_output_dir=str(local_dir),
        artifact_manifest={},
        metrics_summary={},
        error_code=None,
        caveats=[],
    )

    ingested = ingest_remote_job_result(result, workspace_id="remote_test")

    assert ingested["artifact_ids"]
    assert ingested["run_memory"]["current_status"] == "succeeded"
    assert ingested["claims"]
    stored_claims = SQLiteStore().list("claims", workspace_id="remote_test")
    assert stored_claims[0]["status"] in {"supported", "partially_supported"}
