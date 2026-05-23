"""Loop feedback context builder.

Extracts sanitized scalar feedback from a completed iteration for
consumption by the LLM planner in subsequent iterations.
"""

from __future__ import annotations

from typing import Any


def build_feedback_context(
    previous_iteration: Any,
) -> dict[str, Any]:
    """Build a sanitized feedback dict from a previous iteration.

    Extracts only scalar metrics and simple strings. No file paths,
    tensor references, API keys, or raw artifacts.

    Args:
        previous_iteration: An AutonomousLoopIteration object or dict
            with strategy_recommendation, execution_result, and
            claim_gate_decision fields.

    Returns:
        Dict with only scalar/string fields suitable for LLM context.
    """
    strategy = _attr_or_dict(previous_iteration, "strategy_recommendation", {})
    execution = _attr_or_dict(previous_iteration, "execution_result", {})
    payload = execution.get("result_payload") or {}
    claim = _attr_or_dict(previous_iteration, "claim_gate_decision", {})
    iter_id = _attr_or_dict(previous_iteration, "iteration_id", 0)

    loss_before = _safe_float(payload, "reconstruction_loss_before")
    loss_after = _safe_float(payload, "reconstruction_loss_after")

    return {
        "iteration_id": iter_id,
        "previous_action": strategy.get("recommended_action", ""),
        "previous_task_type": strategy.get("task_type", ""),
        "previous_backend": execution.get("backend_id", ""),
        "previous_status": execution.get("status", ""),
        "loss_before": loss_before,
        "loss_after": loss_after,
        "improvement_detected": (
            loss_after is not None
            and loss_before is not None
            and loss_after < loss_before
        ),
        "rollback_count": payload.get("rollback_count", 0),
        "accepted_update_count": payload.get("accepted_update_count", 0),
        "claim_gate_decision": claim.get("decision", ""),
        "failure_mode": _detect_failure_mode(execution, payload),
        "error_message": _truncate(
            (execution.get("errors") or [{}])[0].get("message", ""), 200
        ),
        "pending_backend_switch": execution.get("pending_backend_switch", False),
        "switched_from_backend": execution.get("switched_from_backend", ""),
        "switched_to_backend": execution.get("switched_to_backend", ""),
        "backend_switch_validated": execution.get("backend_switch_validated", False),
    }


def build_recent_results(
    iterations: list[Any],
) -> list[dict[str, Any]]:
    """Build recent_results list from completed iterations.

    Each entry is a feedback context dict. Only iterations with
    execution results are included.
    """
    recent: list[dict[str, Any]] = []
    for it in iterations:
        try:
            ctx = build_feedback_context(it)
            if ctx.get("previous_status"):
                recent.append(ctx)
        except Exception:
            continue
    return recent


def _attr_or_dict(obj: Any, attr: str, default: Any = None) -> Any:
    """Get attribute or dict key, with default."""
    if obj is None:
        return default
    if hasattr(obj, attr):
        val = getattr(obj, attr)
        return val if val is not None else default
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return default


def _safe_float(payload: dict[str, Any], key: str) -> float | None:
    """Safely extract a float value, returning None on failure."""
    val = payload.get(key)
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _truncate(text: str, max_len: int) -> str:
    """Truncate text to max_len characters."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _detect_failure_mode(
    execution: dict[str, Any], payload: dict[str, Any]
) -> str:
    """Classify failure mode from execution result."""
    status = execution.get("status", "")
    if status == "succeeded":
        loss_before = _safe_float(payload, "reconstruction_loss_before")
        loss_after = _safe_float(payload, "reconstruction_loss_after")
        if loss_before is not None and loss_after is not None:
            if loss_after < loss_before:
                return "improvement"
            if payload.get("rollback_count", 0) > 0:
                return "rollback_protected"
            return "no_improvement"
        return "executed"
    if status == "claim_downgraded":
        return "claim_ceiling_too_low"
    if status == "failed":
        return "execution_failure"
    if status == "skipped":
        return "precondition_failure"
    return "unknown"
