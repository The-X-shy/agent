"""Test remote-aware claim ceiling resolver."""

from optiresearch.memory.claim_ceiling_resolver import resolve_claim_ceiling


def test_local_mode_remote_required_handler_returns_needs_followup():
    result = resolve_claim_ceiling(
        handler_id="remote_native_geolens_validation",
        backend_id="deeplens_geolens_geometric",
        execution_target="local",
        handler_remote_required=True,
        supports_remote=True,
    )
    assert result.final_claim_ceiling == "needs_followup"
    assert "remote" in result.limiting_factor.lower()


def test_remote_mode_uses_remote_ceiling():
    result = resolve_claim_ceiling(
        handler_id="remote_native_geolens_validation",
        backend_id="deeplens_geolens_geometric",
        execution_target="remote_wsl",
        handler_remote_required=True,
        supports_remote=True,
        remote_validation_passed=True,
    )
    assert result.final_claim_ceiling == "native_lens_simulation"


def test_remote_validation_failed_returns_needs_followup():
    result = resolve_claim_ceiling(
        handler_id="remote_native_geolens_validation",
        backend_id="deeplens_geolens_geometric",
        execution_target="remote_wsl",
        handler_remote_required=True,
        supports_remote=True,
        remote_validation_passed=False,
    )
    assert result.final_claim_ceiling == "needs_followup"


def test_local_handler_unaffected():
    result = resolve_claim_ceiling(
        handler_id="objective_redesign_simpler_metric",
        backend_id="deeplens_geolens_geometric",
        execution_target="local",
        synthetic_data=True,
        physical_backend=False,
        native_backend=False,
        phase_to_fft_proxy_used=True,
    )
    assert result.final_claim_ceiling == "lightweight_scientific_execution"


def test_no_handler_returns_backend_ceiling():
    result = resolve_claim_ceiling(
        handler_id="",
        backend_id="deeplens_geolens_geometric",
        execution_target="local",
    )
    assert result.final_claim_ceiling == "native_lens_simulation"
