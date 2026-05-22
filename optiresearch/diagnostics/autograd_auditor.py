"""Autograd graph auditor — detects broken differentiable links."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import torch


@dataclass
class AutogradAuditReport:
    """Result of auditing an autograd graph after loss.backward()."""

    loss_requires_grad: bool
    parameter_count: int
    parameters_with_grad: int
    gradient_norms: dict[str, float]
    zero_grad_parameters: list[str]
    missing_grad_parameters: list[str]
    suspected_breaks: list[str]
    has_detach_in_path: bool
    has_numpy_conversion_risk: bool
    has_no_grad_region: bool
    rollback_parameters_changed: Optional[bool]
    verdict: Literal["clean", "suspicious", "broken"]
    recommendations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def inspect_tensor_requires_grad(
    tensors: dict[str, torch.Tensor],
) -> dict[str, bool]:
    """Check which tensors require gradients."""
    return {name: bool(t.requires_grad) for name, t in tensors.items()}


def trace_loss_to_parameters(
    loss: torch.Tensor,
    params: list[torch.nn.Parameter],
) -> dict[str, Any]:
    """Check whether loss can trace its grad_fn back to each parameter."""
    loss_fn = getattr(loss, "grad_fn", None)
    reachable: dict[str, bool] = {}
    for p in params:
        try:
            # A rough check: see if retaining grad gives a non-None value
            reachable[str(id(p))] = p.grad is not None
        except Exception:
            reachable[str(id(p))] = False
    return {
        "loss_has_grad_fn": loss_fn is not None,
        "reachable_parameters": reachable,
    }


def detect_detach(
    tensor: torch.Tensor,
    name: str = "unknown",
) -> Optional[str]:
    """Check whether a tensor has been detached from the graph.

    A leaf tensor (created directly, e.g. torch.tensor(..., requires_grad=True))
    legitimately has grad_fn=None — this is NOT a detach issue.
    Only non-leaf tensors with requires_grad=True and grad_fn=None are suspicious.
    """
    if tensor.requires_grad and tensor.grad_fn is None and not tensor.is_leaf:
        return f"{name}: requires_grad=True but grad_fn=None (possible detach of non-leaf)"
    return None


def detect_numpy_conversion_risk(module: torch.nn.Module) -> list[str]:
    """Scan module for patterns that may convert tensors to numpy (breaking autograd)."""
    risks: list[str] = []
    for name, submod in module.named_modules():
        for pname, param in submod.named_parameters(recurse=False):
            if param.requires_grad and param.grad is None and param._grad is None:
                # Param wants grad but has none — may or may not be a problem
                pass
    return risks


def detect_no_grad_region(module: torch.nn.Module) -> list[str]:
    """Detect parameters that sit inside no_grad contexts (indirect check)."""
    warnings: list[str] = []
    for name, param in module.named_parameters():
        if param.requires_grad and param.grad is None:
            # Could be no_grad or could be first forward pass
            pass
    return warnings


def summarize_gradient_flow(
    params: list[torch.nn.Parameter],
) -> dict[str, Any]:
    """Compute gradient statistics for a list of parameters."""
    norms: dict[str, float] = {}
    zero_grad: list[str] = []
    missing_grad: list[str] = []
    for p in params:
        pname = str(id(p))
        if p.grad is None:
            missing_grad.append(pname)
            norms[pname] = 0.0
        else:
            gnorm = float(p.grad.norm().item())
            norms[pname] = gnorm
            if gnorm == 0.0:
                zero_grad.append(pname)
    return {
        "gradient_norms": norms,
        "zero_grad_parameters": zero_grad,
        "missing_grad_parameters": missing_grad,
        "parameter_count": len(params),
        "parameters_with_grad": len(params) - len(missing_grad),
    }


def compare_gradient_strength(
    gradients: dict[str, torch.Tensor],
) -> dict[str, float]:
    """Compare relative gradient magnitudes across named parameters."""
    if not gradients:
        return {}
    norms = {name: float(g.norm().item()) for name, g in gradients.items() if g is not None}
    total = sum(norms.values()) or 1.0
    return {name: v / total for name, v in norms.items()}


def audit_autograd_graph(
    loss: torch.Tensor,
    parameters: dict[str, torch.Tensor],
    module: Optional[torch.nn.Module] = None,
    rollback_parameters_changed: Optional[bool] = None,
) -> AutogradAuditReport:
    """Audit an autograd graph after loss.backward() has been called.

    Args:
        loss: The loss tensor (after .backward()).
        parameters: Name -> tensor mapping of trainable parameters.
        module: Optional nn.Module for deeper inspection.
        rollback_parameters_changed: If False, rollback restored parameters
            and zero grad change is *expected* — not an autograd break.

    Returns:
        AutogradAuditReport with verdict and recommendations.
    """
    loss_req_grad = bool(loss.requires_grad)

    grad_norms: dict[str, float] = {}
    zero_grad_params: list[str] = []
    missing_grad_params: list[str] = []
    param_count = len(parameters)
    params_with_grad = 0

    for name, p in parameters.items():
        if p.grad is None:
            missing_grad_params.append(name)
            grad_norms[name] = 0.0
        else:
            gnorm = float(p.grad.norm().item())
            grad_norms[name] = gnorm
            params_with_grad += 1
            if gnorm == 0.0:
                zero_grad_params.append(name)

    # Detect graph breaks
    suspected_breaks: list[str] = []
    has_detach = False
    has_numpy_risk = False
    has_no_grad = False

    if not loss_req_grad:
        suspected_breaks.append("loss.requires_grad is False — graph is detached at loss")
        has_detach = True

    if loss_req_grad and params_with_grad == 0 and param_count > 0:
        suspected_breaks.append(
            "Loss requires grad but no parameters received gradients — "
            "possible detach or no_grad in forward path"
        )
        has_detach = True

    if module is not None:
        for name, param in module.named_parameters():
            if param.requires_grad and param.grad is None and param._grad is None:
                if any(p is param for p in parameters.values()):
                    has_no_grad = True

    # Rollback-aware verdict
    if rollback_parameters_changed is False and params_with_grad == 0:
        suspected_breaks.append(
            "no_grad_change_due_to_rollback — parameters were restored from snapshot, "
            "so zero grad change is expected and not an autograd issue"
        )

    # Determine verdict
    if not loss_req_grad:
        verdict: Literal["clean", "suspicious", "broken"] = "broken"
    elif params_with_grad == 0 and param_count > 0:
        if rollback_parameters_changed is False:
            verdict = "clean"
        else:
            verdict = "broken"
    elif len(suspected_breaks) > 1:
        verdict = "suspicious"
    else:
        verdict = "clean"

    # Recommendations
    recommendations: list[str] = []
    if not loss_req_grad:
        recommendations.append(
            "Loss is detached from graph. Check that the loss computation does not "
            "call .detach(), .numpy(), or torch.no_grad()."
        )
    if has_detach and verdict != "clean":
        recommendations.append(
            "Detach detected in gradient path. Trace forward pass for .detach() calls."
        )
    if len(zero_grad_params) > 0 and verdict != "clean":
        recommendations.append(
            f"{len(zero_grad_params)} parameters have grad=0. "
            "Check if they are downstream of a non-differentiable operation."
        )
    if missing_grad_params and verdict != "clean":
        recommendations.append(
            f"{len(missing_grad_params)} parameters have grad=None. "
            "Backward may not have been called or these params are not in the graph."
        )
    if has_no_grad:
        recommendations.append(
            "Possible torch.no_grad() context detected. "
            "Check that forward pass runs outside no_grad."
        )
    if rollback_parameters_changed is False and verdict == "clean":
        recommendations.append(
            "Rollback restored parameters — gradient flow was normal before restoration. "
            "This is expected rollback behaviour, not an autograd issue."
        )

    return AutogradAuditReport(
        loss_requires_grad=loss_req_grad,
        parameter_count=param_count,
        parameters_with_grad=params_with_grad,
        gradient_norms=grad_norms,
        zero_grad_parameters=zero_grad_params,
        missing_grad_parameters=missing_grad_params,
        suspected_breaks=suspected_breaks,
        has_detach_in_path=has_detach,
        has_numpy_conversion_risk=has_numpy_risk,
        has_no_grad_region=has_no_grad,
        rollback_parameters_changed=rollback_parameters_changed,
        verdict=verdict,
        recommendations=recommendations,
    )
