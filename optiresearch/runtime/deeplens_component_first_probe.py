"""DeepLens component-first probe runtime for Phase 62.

Thin wrapper around ``run_surface_optimization_probe()`` that maps
component-level semantics (Fresnel, Binary2Phase, diffractive) to the existing
surface optimization probe engine.  No optimization loop is duplicated.
"""

from __future__ import annotations

from typing import Any

from optiresearch.schemas.component_probe import (
    ComponentProbeResult,
    ComponentProbeSpec,
)

# Component name → surface class name used by the existing surface probe engine.
COMPONENT_TO_SURFACE: dict[str, str] = {
    "fresnel": "Fresnel",
    "binary2phase": "Binary2Phase",
    "diffractive": "Fresnel",
}

# Component name → backend_id for claim gating and evidence tracking.
COMPONENT_TO_BACKEND: dict[str, str] = {
    "fresnel": "deeplens_fresnel_component",
    "binary2phase": "deeplens_binary2phase_component",
    "diffractive": "deeplens_fresnel_component",
}

# Known trainable parameter names per surface class for gradient classification.
SURFACE_ZERO_GRADIENT_CANDIDATES: dict[str, list[str]] = {
    "Fresnel": ["f0"],
    "Binary2Phase": ["d", "order2", "order4", "order6", "order8", "order10", "order12"],
}


def run_deeplens_component_probe(
    spec: ComponentProbeSpec,
) -> ComponentProbeResult:
    """Run a component-level DeepLens optimization probe.

    Maps the component name to a DeepLens surface class, delegates to
    ``run_surface_optimization_probe()``, and translates the result into
    component-specific semantics.

    Parameters
    ----------
    spec:
        Probe specification — component, objective, max_steps, etc.

    Returns
    -------
    ComponentProbeResult
        Structured result with component-level evidence and claim ceiling.
    """
    surface_class = COMPONENT_TO_SURFACE.get(spec.component)
    backend_id = COMPONENT_TO_BACKEND.get(spec.component, "deeplens_fresnel_component")

    if surface_class is None:
        return ComponentProbeResult(
            probe_id=spec.probe_id,
            component=spec.component,
            status="needs_followup",
            backend_id=backend_id,
            error_code="UNKNOWN_COMPONENT",
            error_message=f"Unknown component: {spec.component}",
            evidence_level="diagnostic_evidence",
            claim_ceiling="diagnostic_evidence",
            checked_component_candidates=list(COMPONENT_TO_SURFACE.keys()),
            caveats=[f"No surface class mapping for component: {spec.component}"],
        )

    # Resolve objective to one the surface probe engine understands.
    objective = _map_objective(spec.objective)

    try:
        from optiresearch.schemas.surface_optimization import (
            SurfaceOptimizationProbeSpec,
        )
        from optiresearch.runtime.deeplens_surface_optimization_probe import (
            run_surface_optimization_probe,
        )

        surface_spec = SurfaceOptimizationProbeSpec(
            probe_id=spec.probe_id,
            surface_class=surface_class,
            objective=objective,
            max_steps=spec.max_steps,
            learning_rate=spec.learning_rate,
            device=spec.device,
            save_artifacts=spec.save_artifacts,
        )
        surface_result = run_surface_optimization_probe(surface_spec)
    except ImportError:
        return ComponentProbeResult(
            probe_id=spec.probe_id,
            component=spec.component,
            surface_class=surface_class,
            status="needs_followup",
            backend_id=backend_id,
            error_code="DEEPLENS_COMPONENT_API_UNAVAILABLE",
            error_message="DeepLens component API not available — import failed",
            evidence_level="diagnostic_evidence",
            claim_ceiling="diagnostic_evidence",
            checked_component_candidates=list(COMPONENT_TO_SURFACE.keys()),
            caveats=["DeepLens is not installed or not importable"],
        )

    return _map_surface_to_component_result(spec, surface_result, surface_class, backend_id)


def _map_objective(objective: str) -> str:
    if objective == "parameter_sanity_check":
        return "minimize_phase_variance"
    return objective


def _map_surface_to_component_result(
    spec: ComponentProbeSpec,
    surface_result: Any,
    surface_class: str,
    backend_id: str,
) -> ComponentProbeResult:
    """Translate a SurfaceOptimizationProbeResult into a ComponentProbeResult."""

    sr = surface_result  # shorthand

    trainable_names = list(sr.trainable_params) if sr.trainable_params else []
    zero_grad_params = _classify_zero_gradient_params(sr, surface_class)

    param_count = len(trainable_names)
    params_with_grad = param_count if sr.autograd_graph_exists else 0

    # Determine status with component-level semantics.
    if sr.status == "succeeded" and sr.differentiable:
        status = "succeeded"
        evidence_level = "diagnostic_evidence"
        claim_ceiling = "native_component_optimization"
    elif sr.status == "succeeded":
        status = "needs_followup"
        evidence_level = "diagnostic_evidence"
        claim_ceiling = "diagnostic_evidence"
    elif sr.status == "unsupported":
        status = "structured_unavailable"
        evidence_level = "diagnostic_evidence"
        claim_ceiling = "diagnostic_evidence"
    else:
        status = "failed"
        evidence_level = "diagnostic_evidence"
        claim_ceiling = "diagnostic_evidence"

    return ComponentProbeResult(
        probe_id=spec.probe_id,
        component=spec.component,
        status=status,
        surface_class=surface_class,
        backend_id=backend_id,
        module_path=sr.module_path,
        can_instantiate=sr.can_instantiate,
        has_get_optimizer=sr.has_get_optimizer,
        has_get_optimizer_params=sr.has_get_optimizer_params,
        parameter_count=param_count,
        trainable_param_count=param_count,
        trainable_param_names=trainable_names,
        params_with_grad=params_with_grad,
        zero_gradient_parameters=zero_grad_params,
        differentiable=sr.differentiable,
        autograd_graph_exists=sr.autograd_graph_exists,
        parameters_changed=sr.parameters_changed,
        loss_before=sr.loss_before,
        loss_after=sr.loss_after,
        gradient_norm=sr.gradient_norm,
        parameter_norm_before=sr.parameter_norm_before,
        parameter_norm_after=sr.parameter_norm_after,
        optimizer_class=sr.optimizer_class,
        error_code=sr.error_code,
        error_message=sr.error_message,
        evidence_level=evidence_level,
        claim_ceiling=claim_ceiling,
        checked_component_candidates=list(COMPONENT_TO_SURFACE.keys()),
        caveats=list(sr.caveats) if sr.caveats else [],
        warnings=[],
        metadata={
            **_safe_metadata(sr),
            "surface_status": sr.status,
            "surface_objective": sr.objective,
        },
    )


def _classify_zero_gradient_params(surface_result: Any, surface_class: str) -> list[str]:
    """Identify parameters that have zero or None gradients."""
    candidates = SURFACE_ZERO_GRADIENT_CANDIDATES.get(surface_class, [])
    zero: list[str] = []
    per_param = surface_result.metadata.get("per_parameter_grad_norm", {})
    for name in candidates:
        grad_norm = per_param.get(name)
        if grad_norm is None or grad_norm == 0.0:
            zero.append(name)
    return zero


def _safe_metadata(surface_result: Any) -> dict[str, Any]:
    try:
        return dict(surface_result.metadata)
    except Exception:
        return {}
