"""Tests for native_geolens_rollback_policy module."""

from __future__ import annotations

from optiresearch.runtime.native_geolens_rollback_policy import (
    AcceptanceDecision,
    RollbackPolicy,
    evaluate_native_geolens_update_acceptance,
)


def _good_metrics():
    before = {"mse": 0.5, "psnr": 3.0, "sam": 1.0}
    after = {"mse": 0.4, "psnr": 4.0, "sam": 0.9}
    psf_before = {"centroid_y": 3.0, "centroid_x": 3.0, "width": 10.0}
    psf_after = {"centroid_y": 3.0, "centroid_x": 3.0, "width": 10.0}
    return before, after, psf_before, psf_after


def test_accepts_when_all_metrics_improve():
    before, after, psf_b, psf_a = _good_metrics()
    decision = evaluate_native_geolens_update_acceptance(
        before, after, grad_norm_max=100.0,
        psf_stats_before=psf_b, psf_stats_after=psf_a,
        policy=RollbackPolicy(),
    )
    assert decision.accepted
    assert decision.reasons == []


def test_rejects_when_grad_norm_too_high():
    before, after, psf_b, psf_a = _good_metrics()
    decision = evaluate_native_geolens_update_acceptance(
        before, after, grad_norm_max=6000.0,
        psf_stats_before=psf_b, psf_stats_after=psf_a,
        policy=RollbackPolicy(max_grad_norm=5000.0),
    )
    assert not decision.accepted
    assert any("gradient_norm_too_high" in r for r in decision.reasons)


def test_rejects_when_mse_worse():
    before, after, psf_b, psf_a = _good_metrics()
    after["mse"] = 0.6  # worse than 0.5
    decision = evaluate_native_geolens_update_acceptance(
        before, after, grad_norm_max=100.0,
        psf_stats_before=psf_b, psf_stats_after=psf_a,
        policy=RollbackPolicy(),
    )
    assert not decision.accepted
    assert any("mse_worse" in r for r in decision.reasons)


def test_rejects_when_sam_worse():
    before, after, psf_b, psf_a = _good_metrics()
    after["sam"] = 1.2  # worse than 1.0
    decision = evaluate_native_geolens_update_acceptance(
        before, after, grad_norm_max=100.0,
        psf_stats_before=psf_b, psf_stats_after=psf_a,
        policy=RollbackPolicy(),
    )
    assert not decision.accepted
    assert any("sam_worse" in r for r in decision.reasons)


def test_rejects_when_psnr_worse():
    before, after, psf_b, psf_a = _good_metrics()
    after["psnr"] = 2.0  # worse than 3.0
    decision = evaluate_native_geolens_update_acceptance(
        before, after, grad_norm_max=100.0,
        psf_stats_before=psf_b, psf_stats_after=psf_a,
        policy=RollbackPolicy(),
    )
    assert not decision.accepted
    assert any("psnr_worse" in r for r in decision.reasons)


def test_rejects_when_psf_centroid_shift_high():
    before, after, psf_b, psf_a = _good_metrics()
    psf_a["centroid_y"] = 4.0  # shift of 1.0
    decision = evaluate_native_geolens_update_acceptance(
        before, after, grad_norm_max=100.0,
        psf_stats_before=psf_b, psf_stats_after=psf_a,
        policy=RollbackPolicy(max_psf_centroid_shift=0.5),
    )
    assert not decision.accepted
    assert any("psf_centroid_shift_high" in r for r in decision.reasons)


def test_rejects_when_psf_width_shift_high():
    before, after, psf_b, psf_a = _good_metrics()
    psf_a["width"] = 12.0  # shift of 2.0
    decision = evaluate_native_geolens_update_acceptance(
        before, after, grad_norm_max=100.0,
        psf_stats_before=psf_b, psf_stats_after=psf_a,
        policy=RollbackPolicy(max_psf_width_shift=1.0),
    )
    assert not decision.accepted
    assert any("psf_width_shift_high" in r for r in decision.reasons)


def test_accepts_with_tradeoff_when_score_improves():
    before, after, psf_b, psf_a = _good_metrics()
    # SAM gets worse but MSE improves enough to boost composite score
    after["sam"] = 1.2   # worse
    after["mse"] = 0.01  # much better
    decision = evaluate_native_geolens_update_acceptance(
        before, after, grad_norm_max=100.0,
        psf_stats_before=psf_b, psf_stats_after=psf_a,
        policy=RollbackPolicy(allow_tradeoff=True),
    )
    assert decision.accepted
    assert any("tradeoff_accepted" in r for r in decision.reasons)


def test_all_reasons_recorded():
    before, after, psf_b, psf_a = _good_metrics()
    after["mse"] = 0.6
    after["sam"] = 1.3
    after["psnr"] = 2.0
    psf_a["centroid_y"] = 4.0
    psf_a["width"] = 12.0
    decision = evaluate_native_geolens_update_acceptance(
        before, after, grad_norm_max=6000.0,
        psf_stats_before=psf_b, psf_stats_after=psf_a,
        policy=RollbackPolicy(max_grad_norm=5000.0),
    )
    assert not decision.accepted
    assert len(decision.reasons) >= 5  # all gates triggered


def test_policy_disabled_accepts_all():
    before, after, psf_b, psf_a = _good_metrics()
    after["mse"] = 10.0  # terrible
    after["sam"] = 5.0
    decision = evaluate_native_geolens_update_acceptance(
        before, after, grad_norm_max=9999.0,
        psf_stats_before=psf_b, psf_stats_after=psf_a,
        policy=RollbackPolicy(enabled=False),
    )
    assert decision.accepted
    assert "rollback_disabled" in decision.reasons


def test_stability_score_computed():
    before, after, psf_b, psf_a = _good_metrics()
    decision = evaluate_native_geolens_update_acceptance(
        before, after, grad_norm_max=100.0,
        psf_stats_before=psf_b, psf_stats_after=psf_a,
        policy=RollbackPolicy(),
    )
    assert isinstance(decision.stability_score_before, float)
    assert isinstance(decision.stability_score_after, float)
