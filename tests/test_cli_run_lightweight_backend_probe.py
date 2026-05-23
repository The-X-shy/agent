"""Phase 32: CLI run-lightweight-backend-probe tests."""

from optiresearch.runtime.lightweight_experiments import (
    run_lightweight_backend_probe,
    run_deeplens_geolens_geometric_deep_probe,
)


def test_cli_shallow_probe_api():
    """Test shallow probe via direct function call (CLI equivalent)."""
    result = run_lightweight_backend_probe(backend_id="phase_to_fft_proxy")
    assert result.status == "succeeded"
    assert result.backend_id == "phase_to_fft_proxy"


def test_cli_deep_probe_api():
    """Test deep probe via direct function call (CLI equivalent)."""
    result = run_deeplens_geolens_geometric_deep_probe(
        backend_id="deeplens_geolens_geometric",
    )
    assert result.status in ("succeeded", "failed")
    assert result.backend_id == "deeplens_geolens_geometric"
