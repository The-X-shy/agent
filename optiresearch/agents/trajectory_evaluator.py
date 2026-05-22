"""Trajectory evaluator for autonomous research loops.

Evaluates iteration trajectories to detect progress, stagnation,
repeated failures, and claim ceiling saturation.
"""

from __future__ import annotations

from typing import Any, Optional

from optiresearch.memory.schemas import StrictModel


class TrajectoryEvaluation(StrictModel):
    """Cross-iteration trajectory analysis."""

    improvement_detected: bool = False
    repeated_failure_detected: bool = False
    claim_ceiling_reached: bool = False
    best_iteration: int = -1
    best_metric_value: float = 0.0
    metric_trajectory: list[float] = []
    stop_reason: str = ""
    evaluation: str = ""
    claim_downgraded_count: int = 0
    min_iterations_satisfied: bool = False


def evaluate_trajectory(
    iterations: list["AutonomousLoopIteration"],
    spec: "AutonomousLoopSpec",
    min_iterations_before_stop: int = 2,
    no_improvement_patience: int = 2,
) -> TrajectoryEvaluation:
    """Evaluate trajectory across iterations to determine progress.

    Checks:
    1. improvement_detected — metric improved over time
    2. repeated_failure — >=2 consecutive failed iterations
    3. claim_ceiling_reached — all iterations hit same claim ceiling
    4. best_iteration — index of best metric value

    Phase 29: Uses min_iterations_before_stop to prevent premature
    no_improvement stops, and no_improvement_patience to require
    consecutive non-improving iterations before stopping.

    Args:
        iterations: List of completed AutonomousLoopIteration objects.
        spec: The AutonomousLoopSpec driving the loop.
        min_iterations_before_stop: Minimum iterations before any stop
            reason can be emitted (except repeated_failure, ceiling).
        no_improvement_patience: Consecutive no-improvement iterations
            required before emitting no_improvement stop.

    Returns:
        TrajectoryEvaluation with stop reason and metrics.
    """
    if not iterations:
        return TrajectoryEvaluation(
            stop_reason="no_iterations",
            evaluation="No iterations completed.",
        )

    trajectory: list[float] = []
    consecutive_failures = 0
    max_consecutive_failures = 0
    failure_count = 0
    claim_downgraded_count = 0

    for it in iterations:
        result = it.execution_result or {}
        status = result.get("status", "")
        metrics = result.get("result_payload") or it.metrics_snapshot or {}

        metric_val = _extract_primary_metric(metrics)
        trajectory.append(metric_val if metric_val is not None else 0.0)

        if status in ("failed", "skipped", "unsupported"):
            failure_count += 1
            consecutive_failures += 1
            max_consecutive_failures = max(max_consecutive_failures, consecutive_failures)
        elif status == "claim_downgraded":
            claim_downgraded_count += 1
            consecutive_failures = 0
        else:
            consecutive_failures = 0

    best_idx = _argmin_or_max(trajectory, prefer_lower=True)
    best_val = trajectory[best_idx] if best_idx >= 0 else 0.0
    improvement = best_val < trajectory[0] if len(trajectory) >= 2 else False

    repeated_failure = max_consecutive_failures >= 2

    claim_ceilings: set[str] = set()
    for it in iterations:
        cgd = it.claim_gate_decision or {}
        ceiling = cgd.get("max_allowed_claim")
        if ceiling:
            claim_ceilings.add(str(ceiling))
    ceiling_reached = len(claim_ceilings) == 1 and len(iterations) >= 2

    # Count consecutive no-improvement iterations from most recent
    no_improvement_streak = 0
    best_so_far = float("inf")
    for val in trajectory:
        if val > 0.0:
            if val >= best_so_far:
                no_improvement_streak += 1
            else:
                no_improvement_streak = 0
                best_so_far = val

    min_iter_satisfied = len(iterations) >= min_iterations_before_stop

    stop_reason = ""
    if ceiling_reached:
        stop_reason = "claim_ceiling_reached"
    elif repeated_failure and not improvement:
        stop_reason = "repeated_failure"
    elif len(iterations) >= spec.max_iterations:
        stop_reason = "max_iterations_reached"
    elif (
        min_iter_satisfied
        and not improvement
        and no_improvement_streak >= no_improvement_patience
    ):
        stop_reason = "no_improvement"

    return TrajectoryEvaluation(
        improvement_detected=improvement,
        repeated_failure_detected=repeated_failure,
        claim_ceiling_reached=ceiling_reached,
        best_iteration=best_idx + 1,
        best_metric_value=best_val,
        metric_trajectory=trajectory,
        stop_reason=stop_reason,
        evaluation=_build_evaluation_text(
            improvement, repeated_failure, ceiling_reached, best_idx, failure_count
        ),
        claim_downgraded_count=claim_downgraded_count,
        min_iterations_satisfied=min_iter_satisfied,
    )


def _extract_primary_metric(payload: Optional[dict[str, Any]]) -> Optional[float]:
    """Extract a primary scalar metric from a result payload.

    Prefers reconstruction_loss_after (lower is better), then loss_after,
    then mse_after.
    """
    if payload is None:
        return None
    for key in ("reconstruction_loss_after", "loss_after", "mse_after"):
        val = payload.get(key)
        if val is not None:
            try:
                return float(val)
            except (ValueError, TypeError):
                pass
    return None


def _argmin_or_max(values: list[float], prefer_lower: bool = True) -> int:
    """Return the index of the best value (lowest or highest)."""
    if not values:
        return -1
    if prefer_lower:
        return min(range(len(values)), key=lambda i: values[i])
    return max(range(len(values)), key=lambda i: values[i])


def _build_evaluation_text(
    improved: bool,
    repeated_fail: bool,
    ceiling: bool,
    best_iter: int,
    fail_count: int,
) -> str:
    parts: list[str] = []
    if improved:
        parts.append(f"Improvement detected (best iteration: {best_iter + 1})")
    if repeated_fail:
        parts.append(f"Repeated failures ({fail_count} total failures)")
    if ceiling:
        parts.append("Claim ceiling reached across all iterations")
    if not parts:
        parts.append("No significant progress detected")
    return "; ".join(parts)
