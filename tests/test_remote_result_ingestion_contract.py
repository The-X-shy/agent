import json

from optiresearch.remote.result_ingestion import (
    RemoteHandlerResult,
    parse_remote_handler_result,
)
from optiresearch.schemas.remote import RemoteJobResult


def _remote_job(tmp_path, metrics):
    local_dir = tmp_path / "remote_jobs" / "remote_job_123"
    local_dir.mkdir(parents=True)
    (local_dir / "result.json").write_text(
        json.dumps(
            {
                "reconstruction_loss_before": 0.4,
                "reconstruction_loss_after": 0.3,
                "improvement_detected": True,
                "accepted_update_count": 1,
            }
        ),
        encoding="utf-8",
    )
    manifest = {"artifacts": [{"path": "result.json", "artifact_type": "manifest"}]}
    return RemoteJobResult(
        job_id="remote_job_123",
        status="succeeded",
        remote_run_id="run_remote_123",
        started_at="2026-05-24T00:00:00Z",
        finished_at="2026-05-24T00:00:01Z",
        command=[
            "/mnt/d/agent/run_agent_python.sh",
            "-m",
            "optiresearch.cli",
            "run-deeplens-native-geolens-hsi-codesign",
            "--remote-job-id",
            "remote_job_1436c05c2c4d6359",
        ],
        stdout_path=str(local_dir / "stdout.txt"),
        stderr_path=str(local_dir / "stderr.txt"),
        remote_output_dir="/mnt/d/agent/workspace/remote_jobs/remote_job_123",
        local_output_dir=str(local_dir),
        artifact_manifest=manifest,
        metrics_summary=metrics,
        error_code=None,
        caveats=[],
    )


def test_parse_remote_handler_result_preserves_boolean_false_and_native_contract(tmp_path):
    result = parse_remote_handler_result(
        _remote_job(
            tmp_path,
            {
                "status": "succeeded",
                "remote_run_id": "run_remote_123",
                "evidence_level": "native_lens_simulation",
                "execution_fidelity": "deeplens_native_geometric",
                "proxy_fallback_used": False,
                "fallback_used": False,
                "deeplens_native_psf_path": "geolens.psf_geometric",
                "full_wave_optics": False,
                "phase_to_fft_proxy_used": False,
            },
        ),
        worker_id="windows_wsl",
    )

    assert isinstance(result, RemoteHandlerResult)
    assert result.status == "succeeded"
    assert result.execution_target == "remote_wsl"
    assert result.remote_validation_passed is True
    assert result.proxy_fallback_used is False
    assert result.full_wave_optics is False
    assert result.phase_to_fft_proxy_used is False
    assert result.deeplens_native_psf_path == "geolens.psf_geometric"


def test_parse_remote_handler_result_missing_required_field_is_structured_failure(tmp_path):
    result = parse_remote_handler_result(
        _remote_job(
            tmp_path,
            {
                "status": "succeeded",
                "remote_run_id": "run_remote_123",
                "evidence_level": "native_lens_simulation",
                "execution_fidelity": "deeplens_native_geometric",
                "proxy_fallback_used": False,
                "full_wave_optics": False,
                "phase_to_fft_proxy_used": False,
            },
        ),
        worker_id="windows_wsl",
    )

    assert result.status == "failed"
    assert result.remote_validation_passed is False
    assert any("deeplens_native_psf_path" in e["message"] for e in result.errors)


def test_parse_remote_handler_result_does_not_treat_fallback_as_native(tmp_path):
    result = parse_remote_handler_result(
        _remote_job(
            tmp_path,
            {
                "status": "succeeded",
                "remote_run_id": "run_remote_123",
                "evidence_level": "native_lens_simulation",
                "execution_fidelity": "adapter_proxy",
                "proxy_fallback_used": True,
                "fallback_used": True,
                "deeplens_native_psf_path": "geolens.psf_geometric",
                "full_wave_optics": False,
                "phase_to_fft_proxy_used": False,
            },
        ),
        worker_id="windows_wsl",
    )

    assert result.status == "failed"
    assert result.remote_validation_passed is False
    assert result.proxy_fallback_used is True
    assert result.evidence_level == "needs_followup"


def test_parse_remote_handler_result_uses_result_payload_run_id_for_structured_failure(tmp_path):
    local_dir = tmp_path / "remote_jobs" / "remote_job_unsupported"
    local_dir.mkdir(parents=True)
    (local_dir / "result.json").write_text(
        json.dumps(
            {
                "run_id": "stable_lens_hsi_unsupported",
                "status": "unsupported",
                "evidence_level": None,
                "execution_fidelity": "deeplens_native_geometric",
                "proxy_fallback_used": False,
                "deeplens_native_psf_path": "geolens.psf_geometric",
                "full_wave_optics": False,
                "phase_to_fft_proxy_used": False,
                "stable_training_succeeded": False,
            }
        ),
        encoding="utf-8",
    )
    remote_job = RemoteJobResult(
        job_id="remote_job_unsupported",
        status="succeeded",
        remote_run_id=None,
        started_at="2026-05-24T00:00:00Z",
        finished_at="2026-05-24T00:00:01Z",
        command=[],
        stdout_path=str(local_dir / "stdout.txt"),
        stderr_path=str(local_dir / "stderr.txt"),
        remote_output_dir="/mnt/d/agent/workspace/remote_jobs/remote_job_unsupported",
        local_output_dir=str(local_dir),
        artifact_manifest={},
        metrics_summary={
            "status": "unsupported",
            "execution_fidelity": "deeplens_native_geometric",
            "proxy_fallback_used": False,
            "deeplens_native_psf_path": "geolens.psf_geometric",
            "full_wave_optics": False,
            "phase_to_fft_proxy_used": False,
            "evidence_level": None,
        },
        error_code=None,
        caveats=[],
    )

    result = parse_remote_handler_result(remote_job, worker_id="windows_wsl")

    assert result.status == "failed"
    assert result.run_id == "stable_lens_hsi_unsupported"
    assert result.remote_validation_passed is False
    assert any(error["type"] == "REMOTE_HANDLER_STATUS_NOT_SUCCEEDED" for error in result.errors)
