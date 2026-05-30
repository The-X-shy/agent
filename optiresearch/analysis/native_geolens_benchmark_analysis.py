"""Benchmark analysis utilities for native GeoLens stability reproducibility.

Pure functions operating on lists of NativeGeoLensBenchmarkConfigResult.
"""

from __future__ import annotations

import statistics
from typing import Any


def compute_improvement_rates(
    results: list[Any],
) -> dict[str, float]:
    """Compute per-metric and all-metrics improvement rates."""
    completed = [r for r in results if r.status == "succeeded"]
    n = len(completed)
    if n == 0:
        return {
            "all_metrics_improved_rate": 0.0,
            "mse_improved_rate": 0.0,
            "psnr_improved_rate": 0.0,
            "sam_improved_rate": 0.0,
            "completed_count": 0,
        }

    mse_ok = sum(1 for r in completed if getattr(r, "mse_improved", False))
    psnr_ok = sum(1 for r in completed if getattr(r, "psnr_improved", False))
    sam_ok = sum(1 for r in completed if getattr(r, "sam_improved", False))
    all_ok = sum(
        1 for r in completed
        if getattr(r, "mse_improved", False)
        and getattr(r, "psnr_improved", False)
        and getattr(r, "sam_improved", False)
    )

    return {
        "all_metrics_improved_rate": all_ok / n,
        "mse_improved_rate": mse_ok / n,
        "psnr_improved_rate": psnr_ok / n,
        "sam_improved_rate": sam_ok / n,
        "all_metrics_improved_count": all_ok,
        "completed_count": n,
    }


def compute_metric_statistics(
    results: list[Any],
) -> dict[str, Any]:
    """Compute mean/std of metric deltas across completed results."""
    completed = [r for r in results if r.status == "succeeded"]
    if not completed:
        return {}

    def _safe_list(attr: str) -> list[float]:
        return [float(getattr(r, attr)) for r in completed if getattr(r, attr) is not None]

    stats: dict[str, Any] = {}
    for attr in ("mse_delta", "psnr_delta", "sam_delta", "grad_norm_max"):
        vals = _safe_list(attr)
        if vals:
            stats[f"mean_{attr}"] = statistics.mean(vals)
            stats[f"std_{attr}"] = statistics.stdev(vals) if len(vals) >= 2 else 0.0
    return stats


def identify_best_config(
    results: list[Any],
) -> str:
    """Identify best config by stability_score, penalized if SAM worsened."""
    completed = [r for r in results if r.status == "succeeded"]
    if not completed:
        return ""

    def _score(r: Any) -> float:
        base = float(getattr(r, "stability_score", 0.0) or 0.0)
        if getattr(r, "sam_improved", False):
            base += 0.5
        if getattr(r, "parameter_changed", False):
            base += 0.3
        return base

    best = max(completed, key=_score)
    return getattr(best, "config_id", "")


def identify_robust_config_family(
    results: list[Any],
    min_seeds: int = 3,
    min_all_improved_rate: float = 0.6,
    min_sam_rate: float = 0.6,
) -> str:
    """Identify config families with reproducible improvement across seeds.

    Groups by (steps, spectral_angle_weight, grad_clip_norm) and checks
    whether each family meets reproducibility thresholds.
    """
    completed = [r for r in results if r.status == "succeeded"]
    if not completed:
        return "insufficient_reproducibility: no completed configs"

    groups: dict[str, list[Any]] = {}
    for r in completed:
        key = f"s{r.steps}_w{getattr(r, 'spectral_angle_weight', 0.2)}_c{getattr(r, 'grad_clip_norm', 1000)}"
        groups.setdefault(key, []).append(r)

    robust: list[str] = []
    for key, group in groups.items():
        unique_seeds = len({r.seed for r in group})
        if unique_seeds < min_seeds:
            continue
        rates = compute_improvement_rates(group)
        if (
            rates.get("all_metrics_improved_rate", 0) >= min_all_improved_rate
            and rates.get("sam_improved_rate", 0) >= min_sam_rate
        ):
            robust.append(f"{key} (seeds={unique_seeds}, rate={rates['all_metrics_improved_rate']:.2f})")

    if robust:
        return "; ".join(robust)
    return "insufficient_reproducibility: no config family meets thresholds"


def generate_claim_recommendation(
    rates: dict[str, float],
    seed_count: int,
) -> tuple[str, str]:
    """Generate claim recommendation and safe wording based on benchmark stats.

    Returns (recommendation, safe_wording).
    """
    all_rate = rates.get("all_metrics_improved_rate", 0.0)
    sam_rate = rates.get("sam_improved_rate", 0.0)

    if seed_count < 3:
        return (
            "insufficient_reproducibility",
            "Insufficient seeds for reproducibility claim; results limited to "
            f"{seed_count} seed(s). At least 3 seeds required for benchmark confidence.",
        )

    if all_rate >= 0.6 and sam_rate >= 0.6:
        return (
            "reproducible_synthetic_stability",
            f"Native GeoLens geometric synthetic HSI optimization shows "
            f"reproducible multi-metric improvement across {seed_count} seeds "
            f"(all-metrics rate: {all_rate:.0%}, SAM rate: {sam_rate:.0%}). "
            f"Reproducibility is demonstrated within the tested synthetic benchmark "
            f"configurations; results do not extend to real HSI or wave-optics settings.",
        )

    if all_rate >= 0.4:
        return (
            "limited_evidence",
            f"Native GeoLens geometric synthetic HSI optimization shows "
            f"partial reproducibility (all-metrics rate: {all_rate:.0%}, "
            f"{seed_count} seeds). Results are limited to tested "
            f"configurations and do not yet demonstrate robust reproducibility.",
        )

    return (
        "limited_evidence",
        f"Native GeoLens geometric synthetic HSI optimization shows "
        f"low reproducibility (all-metrics rate: {all_rate:.0%}). "
        f"Further stabilization is needed before claiming reproducible improvement.",
    )


def compute_rollback_statistics(
    results: list[Any],
) -> dict[str, Any]:
    """Compute rollback rate and most common reasons."""
    completed = [r for r in results if r.status == "succeeded"]
    n = len(completed)
    if n == 0:
        return {"rollback_rate": 0.0, "common_reasons": []}

    rollback_count = sum(1 for r in completed if getattr(r, "rollback_count", 0) > 0)
    from collections import Counter

    reason_counts: Counter = Counter()
    for r in completed:
        for reason in getattr(r, "rollback_reasons", []):
            reason_counts[reason.split(":")[0]] += 1

    return {
        "rollback_rate": rollback_count / n,
        "common_reasons": [r for r, _ in reason_counts.most_common(5)],
    }


def compute_tradeoff_labels(
    results: list[Any],
) -> list[str]:
    """Extract unique tradeoff summary labels from results."""
    completed = [r for r in results if r.status == "succeeded"]
    seen: set[str] = set()
    labels: list[str] = []
    for r in completed:
        label = getattr(r, "metric_tradeoff_summary", "")
        if label and label not in seen:
            seen.add(label)
            labels.append(label)
    return labels


def compute_full_grid_improvement_rates(
    results: list[Any],
) -> dict[str, float]:
    """Compute improvement rates across ALL configs (full grid)."""
    total = len(results)
    if total == 0:
        return {"all_metrics_improved_rate_full_grid": 0.0}

    all_ok = sum(
        1 for r in results
        if getattr(r, "mse_improved", False)
        and getattr(r, "psnr_improved", False)
        and getattr(r, "sam_improved", False)
    )
    return {"all_metrics_improved_rate_full_grid": all_ok / total,
            "all_metrics_improved_count_full_grid": all_ok}


def aggregate_config_results(
    results: list[Any],
    benchmark_id: str = "",
) -> dict[str, Any]:
    """Aggregate all config results into a benchmark summary dict.

    Produces both completed-only and full-grid statistics.
    """
    obj_results = results
    total = len(results)
    completed = [r for r in results if getattr(r, "status", "") == "succeeded"]
    unsupported = [r for r in results if getattr(r, "status", "") == "unsupported"]
    failed = [r for r in results if getattr(r, "status", "") == "failed"]

    # Completed-only rates
    rates_completed = compute_improvement_rates(obj_results)
    stats = compute_metric_statistics(obj_results)
    rollback = compute_rollback_statistics(obj_results)

    # Full-grid rates
    rates_full = compute_full_grid_improvement_rates(obj_results)

    completion_rate = len(completed) / total if total > 0 else 0.0
    seeds = sorted({getattr(r, "seed", -1) for r in results if getattr(r, "seed", -1) >= 0})
    best = identify_best_config(obj_results)
    robust = identify_robust_config_family(obj_results)

    # Claim recommendation considers completion coverage
    rec, wording = _generate_claim_recommendation_with_coverage(
        rates_completed, rates_full, completion_rate, len(seeds),
        len(unsupported), len(failed),
    )

    blocked = [
        "real HSI performance validation",
        "full wave-optics HSI co-design",
        "real camera validation",
        "production-ready lens design",
        "guaranteed monotonic improvement across all metrics",
    ]
    if completion_rate < 0.8:
        blocked.append("full-grid reproducibility claim (completion < 80%)")
    if rates_full.get("all_metrics_improved_rate_full_grid", 0) < 0.5:
        blocked.append("full-grid all-metrics improvement claim")

    return {
        "benchmark_id": benchmark_id,
        "config_count": total,
        "completed_count": len(completed),
        "unsupported_count": len(unsupported),
        "failed_count": len(failed),
        "completion_rate": completion_rate,
        "seed_count": len(seeds),
        "all_metrics_improved_count": rates_completed.get("all_metrics_improved_count", 0),
        "all_metrics_improved_rate": rates_completed["all_metrics_improved_rate"],
        "all_metrics_improved_rate_full_grid": rates_full["all_metrics_improved_rate_full_grid"],
        "mse_improved_rate": rates_completed["mse_improved_rate"],
        "psnr_improved_rate": rates_completed["psnr_improved_rate"],
        "sam_improved_rate": rates_completed["sam_improved_rate"],
        "mean_mse_delta": stats.get("mean_mse_delta"),
        "std_mse_delta": stats.get("std_mse_delta"),
        "mean_psnr_delta": stats.get("mean_psnr_delta"),
        "std_psnr_delta": stats.get("std_psnr_delta"),
        "mean_sam_delta": stats.get("mean_sam_delta"),
        "std_sam_delta": stats.get("std_sam_delta"),
        "mean_grad_norm_max": stats.get("mean_grad_norm_max"),
        "rollback_rate": rollback["rollback_rate"],
        "best_config_id": best,
        "robust_config_family": robust,
        "claim_recommendation": rec,
        "safe_wording": wording,
        "blocked_claims": blocked,
    }


def _generate_claim_recommendation_with_coverage(
    rates_completed: dict[str, float],
    rates_full: dict[str, float],
    completion_rate: float,
    seed_count: int,
    unsupported_count: int,
    failed_count: int,
) -> tuple[str, str]:
    """Generate claim recommendation accounting for completion coverage."""
    all_rate_completed = rates_completed.get("all_metrics_improved_rate", 0.0)
    all_rate_full = rates_full.get("all_metrics_improved_rate_full_grid", 0.0)

    if seed_count < 3:
        return (
            "insufficient_reproducibility",
            f"Insufficient seeds ({seed_count}) for reproducibility claim.",
        )

    if completion_rate < 0.8:
        qualifier = f"among completed configurations ({int(completion_rate * 100)}% of grid)"
        if unsupported_count > 0:
            qualifier += f"; {unsupported_count} configs did not improve metrics"
        if all_rate_completed >= 0.8:
            return (
                "completed_configs_only",
                f"Native GeoLens geometric synthetic HSI optimization shows "
                f"reproducible multi-metric improvement {qualifier}. "
                f"Full-grid reproducibility ({all_rate_full:.0%}) is limited by "
                f"non-improving configs. Results do not extend to real HSI or wave-optics.",
            )
        return (
            "limited_evidence",
            f"Improvement is limited {qualifier}. Further stabilization needed.",
        )

    if all_rate_full >= 0.5 and all_rate_completed >= 0.8:
        return (
            "reproducible_synthetic_stability",
            f"Native GeoLens geometric synthetic HSI optimization shows "
            f"reproducible multi-metric improvement (full-grid: {all_rate_full:.0%}, "
            f"completed: {all_rate_completed:.0%}) across {seed_count} seeds. "
            f"Reproducibility demonstrated within tested benchmark. "
            f"Results do not extend to real HSI or wave-optics.",
        )

    return generate_claim_recommendation(rates_completed, seed_count)
