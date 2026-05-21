"""Mock tests for remote native HSI co-design."""

from optiresearch.remote.command_allowlist import validate_remote_command


def test_allowlist_accepts_native_hsi_codesign():
    cmd = [
        "python", "-m", "optiresearch.cli", "run-native-hsi-codesign",
        "--optical-component", "Fresnel",
        "--objective", "minimize_hsi_proxy_loss",
        "--max-steps", "3",
        "--learning-rate", "0.001",
        "--device", "cpu",
        "--bands", "31",
        "--image-size", "32",
        "--psf-size", "16",
        "--remote-job-id", "test-job-123",
    ]
    result = validate_remote_command(cmd)
    assert result["allowed"] is True
    assert result["cli_command"] == "run-native-hsi-codesign"


def test_remote_hsi_codesign_job_spec():
    from optiresearch.runtime.remote_jobs import _job
    job = _job(
        "native_hsi_codesign",
        objective="Test HSI co-design",
        cli_args={"optical_component": "Fresnel", "objective": "minimize_hsi_proxy_loss"},
        timeout_seconds=1800,
        expected_outputs=["result.json"],
    )
    assert job.job_type == "native_hsi_codesign"
    assert "result.json" in job.expected_outputs


def test_remote_claim_scope_for_hsi_codesign():
    from optiresearch.runtime.remote_jobs import _remote_claim_scope
    scope = _remote_claim_scope("native_hsi_codesign")
    assert "optical-HSI" in scope.lower() or "hsi" in scope.lower()


def test_remote_backend_capability_for_hsi_codesign():
    from optiresearch.runtime.remote_jobs import _remote_backend_capability_level
    level = _remote_backend_capability_level("native_hsi_codesign")
    assert level == "native_component"
