"""Autograd and gradient-flow diagnostics."""

from optiresearch.diagnostics.autograd_auditor import (
    AutogradAuditReport,
    audit_autograd_graph,
    compare_gradient_strength,
    detect_detach,
    detect_no_grad_region,
    detect_numpy_conversion_risk,
    inspect_tensor_requires_grad,
    summarize_gradient_flow,
    trace_loss_to_parameters,
)

__all__ = [
    "AutogradAuditReport",
    "audit_autograd_graph",
    "compare_gradient_strength",
    "detect_detach",
    "detect_no_grad_region",
    "detect_numpy_conversion_risk",
    "inspect_tensor_requires_grad",
    "summarize_gradient_flow",
    "trace_loss_to_parameters",
]
