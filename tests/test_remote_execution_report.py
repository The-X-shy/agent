import json

from optiresearch.reports.remote_execution import export_remote_execution_report


def test_remote_execution_report_contains_required_sections(tmp_path):
    job_dir = tmp_path / "remote_jobs" / "job_1"
    job_dir.mkdir(parents=True)
    (job_dir / "remote_job_result.json").write_text(
        json.dumps(
            {
                "job_id": "job_1",
                "status": "succeeded",
                "remote_run_id": "remote_1",
                "error_code": None,
                "metrics_summary": {"fallback_used": False},
                "caveats": ["black-box, not native differentiable optimization"],
                "local_output_dir": str(job_dir),
            }
        ),
        encoding="utf-8",
    )
    (job_dir / "ingestion_summary.json").write_text(
        json.dumps({"artifact_ids": ["artifact_1"], "claims": [{"claim_id": "claim_1", "status": "supported"}]}),
        encoding="utf-8",
    )

    path = export_remote_execution_report("job_1", remote_jobs_root=tmp_path / "remote_jobs")

    text = path.read_text(encoding="utf-8")
    assert "Remote Execution Report" in text
    assert "job_1" in text
    assert "artifact_1" in text
    assert "not native differentiable optimization" in text
