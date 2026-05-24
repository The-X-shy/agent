"""Test ClaimCeilingResolver."""

from optiresearch.memory.claim_ceiling_resolver import (
    resolve_claim_ceiling,
    ClaimCeilingResult,
    _evidence_rank,
)


def test_resolver_handler_ceiling_takes_priority():
    result = resolve_claim_ceiling(
        handler_id="objective_redesign_simpler_metric",
        backend_id="deeplens_geolens_geometric",
        dataset="synthetic",
        execution_fidelity="lightweight_proxy",
    )
    assert result.handler_claim_ceiling == "lightweight_scientific_execution"
    assert result.backend_claim_ceiling == "native_lens_simulation"
    assert result.final_claim_ceiling == "lightweight_scientific_execution"


def test_resolver_unknown_handler_returns_needs_followup():
    result = resolve_claim_ceiling(
        handler_id="nonexistent_handler",
        backend_id="deeplens_geolens_geometric",
    )
    assert result.final_claim_ceiling == "needs_followup"
    assert result.warnings


def test_resolver_empty_handler_falls_back_to_backend():
    result = resolve_claim_ceiling(
        handler_id="",
        backend_id="deeplens_geolens_geometric",
    )
    assert result.final_claim_ceiling == "native_lens_simulation"


def test_resolver_synthetic_data_limits_ceiling():
    result = resolve_claim_ceiling(
        handler_id="objective_redesign_simpler_metric",
        backend_id="deeplens_geolens_geometric",
        synthetic_data=True,
        real_data=False,
    )
    rank = _evidence_rank(result.final_claim_ceiling)
    assert rank <= _evidence_rank("lightweight_scientific_execution")


def test_resolver_no_physical_backend_limits_ceiling():
    result = resolve_claim_ceiling(
        handler_id="objective_redesign_simpler_metric",
        backend_id="deeplens_geolens_geometric",
        physical_backend=False,
        native_backend=False,
    )
    assert result.final_claim_ceiling == "lightweight_scientific_execution"


def test_resolver_proxy_fallback_limits_ceiling():
    result = resolve_claim_ceiling(
        handler_id="objective_redesign_simpler_metric",
        backend_id="deeplens_geolens_geometric",
        proxy_fallback_used=True,
    )
    assert result.final_claim_ceiling == "lightweight_scientific_execution"


def test_resolver_has_downgrade_reasons():
    result = resolve_claim_ceiling(
        handler_id="objective_redesign_simpler_metric",
        backend_id="deeplens_geolens_geometric",
        physical_backend=False,
        native_backend=False,
        synthetic_data=True,
        phase_to_fft_proxy_used=True,
    )
    assert len(result.downgrade_reasons) >= 2


def test_resolver_has_limiting_factor():
    result = resolve_claim_ceiling(
        handler_id="objective_redesign_simpler_metric",
        backend_id="deeplens_geolens_geometric",
    )
    assert result.limiting_factor


def test_evidence_rank_ordering():
    assert _evidence_rank("report_only") < _evidence_rank("lightweight_scientific_execution")
    assert _evidence_rank("lightweight_scientific_execution") < _evidence_rank("native_lens_simulation")
    assert _evidence_rank("native_lens_simulation") < _evidence_rank("real_hsi")
    assert _evidence_rank("unsupported") == 0
