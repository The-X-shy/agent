"""Multi-metric rollback acceptance policy for native GeoLens HSI stabilization.

Gates each optical update step on independent checks:
  - gradient norm below threshold
  - MSE not worse
  - PSNR not worse
  - SAM not worse
  - PSF centroid / width shifts within bounds

Supports optional tradeoff scoring where an update can be accepted despite
individual metric regressions if the composite stability score improves.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RollbackPolicy:
    """Per-step acceptance policy for optical parameter updates."""

    enabled: bool = True
    accept_if_mse_not_worse: bool = True
    accept_if_psnr_not_worse: bool = True
    accept_if_sam_not_worse: bool = True
    max_grad_norm: float = 5000.0
    max_psf_centroid_shift: float = 0.5
    max_psf_width_shift: float = 1.0
    mse_tolerance: float = 0.0
    psnr_tolerance: float = 0.0
    sam_tolerance: float = 0.0
    allow_tradeoff: bool = False
    tradeoff_score_weights: dict[str, float] = field(default_factory=lambda: {
        "mse": 1.0, "psnr": 1.0, "sam": 1.0,
    })


@dataclass
class AcceptanceDecision:
    """Result of evaluating whether to accept an optical update step."""

    accepted: bool
    reasons: list[str] = field(default_factory=list)
    stability_score_before: float = 0.0
    stability_score_after: float = 0.0


def evaluate_native_geolens_update_acceptance(
    metrics_before: dict[str, float],
    metrics_after: dict[str, float],
    grad_norm_max: float,
    psf_stats_before: dict[str, float],
    psf_stats_after: dict[str, float],
    policy: RollbackPolicy,
) -> AcceptanceDecision:
    """Evaluate whether to accept an optical parameter update step.

    Args:
        metrics_before: {"mse": ..., "psnr": ..., "sam": ...} before update
        metrics_after:  {"mse": ..., "psnr": ..., "sam": ...} after update
        grad_norm_max: Max optical gradient norm from this step
        psf_stats_before: {"centroid_y": ..., "centroid_x": ..., "width": ...}
        psf_stats_after:  {"centroid_y": ..., "centroid_x": ..., "width": ...}
        policy: Acceptance policy configuration

    Returns:
        AcceptanceDecision with accepted flag, reasons, and stability scores
    """
    reasons: list[str] = []

    if not policy.enabled:
        return AcceptanceDecision(accepted=True, reasons=["rollback_disabled"])

    # Gradient norm gate
    if grad_norm_max > policy.max_grad_norm:
        reasons.append(
            f"gradient_norm_too_high: {grad_norm_max:.1f} > {policy.max_grad_norm:.1f}"
        )

    # MSE gate
    if policy.accept_if_mse_not_worse:
        mse_before = metrics_before.get("mse", 0.0)
        mse_after = metrics_after.get("mse", 0.0)
        mse_delta = mse_after - mse_before
        if mse_delta > policy.mse_tolerance:
            reasons.append(f"mse_worse: +{mse_delta:.6f}")

    # PSNR gate
    if policy.accept_if_psnr_not_worse:
        psnr_before = metrics_before.get("psnr", 0.0)
        psnr_after = metrics_after.get("psnr", 0.0)
        psnr_delta = psnr_after - psnr_before
        if psnr_delta < -policy.psnr_tolerance:
            reasons.append(f"psnr_worse: {psnr_delta:.4f}")

    # SAM gate
    if policy.accept_if_sam_not_worse:
        sam_before = metrics_before.get("sam", 0.0)
        sam_after = metrics_after.get("sam", 0.0)
        sam_delta = sam_after - sam_before
        if sam_delta > policy.sam_tolerance:
            reasons.append(f"sam_worse: +{sam_delta:.6f}")

    # PSF centroid shift gate
    cy_before = psf_stats_before.get("centroid_y", 0.0)
    cx_before = psf_stats_before.get("centroid_x", 0.0)
    cy_after = psf_stats_after.get("centroid_y", 0.0)
    cx_after = psf_stats_after.get("centroid_x", 0.0)
    centroid_shift = abs(cy_after - cy_before) + abs(cx_after - cx_before)
    if centroid_shift > policy.max_psf_centroid_shift:
        reasons.append(
            f"psf_centroid_shift_high: {centroid_shift:.4f} > {policy.max_psf_centroid_shift:.4f}"
        )

    # PSF width shift gate
    width_before = psf_stats_before.get("width", 0.0)
    width_after = psf_stats_after.get("width", 0.0)
    width_shift = abs(width_after - width_before)
    if width_shift > policy.max_psf_width_shift:
        reasons.append(
            f"psf_width_shift_high: {width_shift:.4f} > {policy.max_psf_width_shift:.4f}"
        )

    score_before = _compute_stability_score(metrics_before, policy.tradeoff_score_weights)
    score_after = _compute_stability_score(metrics_after, policy.tradeoff_score_weights)

    if not reasons:
        accepted = True
    elif policy.allow_tradeoff and score_after > score_before:
        reasons.append(
            f"tradeoff_accepted: score {score_before:.3f} -> {score_after:.3f}"
        )
        accepted = True
    else:
        accepted = False

    return AcceptanceDecision(
        accepted=accepted,
        reasons=reasons,
        stability_score_before=score_before,
        stability_score_after=score_after,
    )


def _compute_stability_score(
    metrics: dict[str, float], weights: dict[str, float]
) -> float:
    """Lower MSE is better, higher PSNR is better, lower SAM is better."""
    score = 0.0
    if "mse" in metrics and weights.get("mse", 0) > 0:
        score -= weights["mse"] * metrics["mse"]
    if "psnr" in metrics and weights.get("psnr", 0) > 0:
        score += weights["psnr"] * metrics["psnr"] * 0.1
    if "sam" in metrics and weights.get("sam", 0) > 0:
        score -= weights["sam"] * metrics["sam"]
    return score
