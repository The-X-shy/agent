"""Mock tests for remote native HSI reconstruction co-design."""

from optiresearch.remote.command_allowlist import validate_remote_command


def test_allowlist_accepts_reconstruction_codesign():
    cmd = [
        "python", "-m", "optiresearch.cli", "run-native-hsi-reconstruction-codesign",
        "--optical-component", "Fresnel",
        "--reconstructor", "differentiable_linear",
        "--max-steps", "5",
        "--optical-lr", "0.001",
        "--recon-lr", "0.001",
        "--device", "cpu",
        "--bands", "31",
        "--image-size", "32",
        "--psf-size", "16",
        "--remote-job-id", "test-job-456",
    ]
    result = validate_remote_command(cmd)
    assert result["allowed"] is True
    assert result["cli_command"] == "run-native-hsi-reconstruction-codesign"


def test_remote_recon_job_spec():
    from optiresearch.runtime.remote_jobs import _job
    job = _job(
        "native_hsi_reconstruction_codesign",
        objective="Test HSI recon co-design",
        cli_args={"optical_component": "Fresnel", "reconstructor": "tiny_cnn"},
        timeout_seconds=1800,
        expected_outputs=["result.json"],
    )
    assert job.job_type == "native_hsi_reconstruction_codesign"


def test_remote_claim_scope_for_reconstruction():
    from optiresearch.runtime.remote_jobs import _remote_claim_scope
    scope = _remote_claim_scope("native_hsi_reconstruction_codesign")
    assert "reconstruction" in scope.lower()


def test_remote_backend_capability_for_reconstruction():
    from optiresearch.runtime.remote_jobs import _remote_backend_capability_level
    level = _remote_backend_capability_level("native_hsi_reconstruction_codesign")
    assert level == "native_component"
