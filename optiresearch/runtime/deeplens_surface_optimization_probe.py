"""Surface-level DeepLens native optimization probes for Phase 19B."""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from optiresearch.schemas.surface_optimization import SurfaceOptimizationProbeResult, SurfaceOptimizationProbeSpec


SURFACE_MODULES: dict[str, str] = {
    "Fresnel": "deeplens.diffractive_surface.fresnel",
    "Binary2": "deeplens.diffractive_surface.binary2",
    "Zernike": "deeplens.diffractive_surface.zernike",
    "Grating": "deeplens.diffractive_surface.grating",
    "Pixel2D": "deeplens.diffractive_surface.pixel2d",
    "ThinLens": "deeplens.diffractive_surface.thinlens",
    "Binary2Phase": "deeplens.phase_surface.binary2",
    "CubicPhase": "deeplens.phase_surface.cubic",
    "ZernikePhase": "deeplens.phase_surface.zernike",
    "PolyPhase": "deeplens.phase_surface.poly",
    "GratingPhase": "deeplens.phase_surface.grating",
    "FresnelPhase": "deeplens.phase_surface.fresnel",
    "NURBSPhase": "deeplens.phase_surface.nurbs",
    "QPhase": "deeplens.phase_surface.qphase",
    "VortexPhase": "deeplens.phase_surface.vortex",
}

SURFACE_TRAINABLE_NAMES: dict[str, list[str]] = {
    "Fresnel": ["f0"],
    "ThinLens": ["f0"],
    "Binary2": ["alpha2", "alpha4", "alpha6", "alpha8", "alpha10"],
    "Zernike": ["z_coeff"],
    "Grating": ["theta", "alpha"],
    "Pixel2D": ["phase_map"],
    "Binary2Phase": ["d", "order2", "order4", "order6", "order8", "order10", "order12"],
    "FresnelPhase": ["f0"],
    "CubicPhase": ["coeff_x3", "coeff_y3", "coeff_x2y", "coeff_xy2", "coeff_x3y", "coeff_xy3"],
    "ZernikePhase": ["z_coeff"],
    "PolyPhase": ["order2", "order3", "order4", "order5", "order6", "order7"],
    "GratingPhase": ["theta", "alpha"],
    "NURBSPhase": ["control_points", "weights"],
    "QPhase": ["coeff_x4", "coeff_y4", "coeff_x3y", "coeff_xy3", "coeff_x2y2"],
    "VortexPhase": ["f0"],
}


def run_surface_optimization_probe(spec: SurfaceOptimizationProbeSpec) -> SurfaceOptimizationProbeResult:
    """Run a native optimization probe on one DeepLens surface class."""
    metadata: dict[str, Any] = {"optimizer_step_executed": False}
    caveats: list[str] = []

    try:
        surface_cls, module_path, import_error = _import_surface_class(spec.surface_class)
    except Exception as exc:
        return _unsupported(spec, "IMPORT_FAILED", str(exc), metadata, caveats)
    if surface_cls is None:
        return _unsupported(spec, "IMPORT_FAILED", import_error or "surface class not importable", metadata, caveats)

    try:
        surface = _instantiate_surface(spec.surface_class, surface_cls, spec.device)
    except Exception as exc:
        return _unsupported(spec, "INSTANTIATION_FAILED", str(exc), metadata, caveats, module_path=module_path)

    try:
        optimizer = _get_optimizer(surface, spec.surface_class, spec.learning_rate)
    except Exception as exc:
        return _unsupported(spec, "OPTIMIZER_UNAVAILABLE", str(exc), metadata, caveats, module_path=module_path)

    trainable_params = _trainable_parameter_names(surface, optimizer, spec.surface_class)
    requires_grad_true = _requires_grad_true(optimizer)
    parameter_before = _parameter_snapshot(surface, optimizer, trainable_params)
    metadata["parameter_before"] = parameter_before
    metadata["requires_grad_true"] = requires_grad_true

    phase_before_np: np.ndarray | None = None
    phase_after_np: np.ndarray | None = None
    target_phase = None
    loss_trace: list[dict[str, Any]] = []
    gradient_norm: float | None = None
    loss_before_val: float | None = None
    loss_after_val: float | None = None

    try:
        for step in range(spec.max_steps):
            optimizer.zero_grad()
            phase = _surface_phase(surface, spec.surface_class, spec.device)
            if step == 0:
                phase_before_np = _to_numpy(phase)
            loss, target_phase = _phase_loss(phase, spec.objective, target_phase)
            if step == 0:
                loss_before_val = float(loss.detach().cpu().item())
            loss.backward()
            gradient_norm = _gradient_norm(optimizer)
            optimizer.step()
            metadata["optimizer_step_executed"] = True
            loss_trace.append(
                {
                    "step": step,
                    "loss": float(loss.detach().cpu().item()),
                    "gradient_norm": gradient_norm,
                }
            )
        phase_after = _surface_phase(surface, spec.surface_class, spec.device)
        loss_after, _ = _phase_loss(phase_after, spec.objective, target_phase)
        loss_after_val = float(loss_after.detach().cpu().item())
        phase_after_np = _to_numpy(phase_after)
    except Exception as exc:
        metadata["parameter_after"] = _parameter_snapshot(surface, optimizer, trainable_params)
        result = SurfaceOptimizationProbeResult(
            probe_id=spec.probe_id,
            status="failed",
            surface_class=spec.surface_class,
            objective=spec.objective,
            module_path=module_path,
            can_instantiate=True,
            has_get_optimizer_params=callable(getattr(surface, "get_optimizer_params", None)),
            has_get_optimizer=callable(getattr(surface, "get_optimizer", None)),
            trainable_params=trainable_params,
            differentiable=False,
            autograd_graph_exists=False,
            loss_before=loss_before_val,
            loss_after=loss_after_val,
            gradient_norm=gradient_norm,
            parameters_changed=False,
            optimizer_class=type(optimizer).__name__,
            error_code="SURFACE_PROBE_FAILED",
            error_message=str(exc),
            caveats=[*caveats, str(exc)],
            metadata=metadata,
        )
        if spec.save_artifacts:
            result.artifact_paths = _save_artifacts(spec, result, loss_trace, phase_before_np, phase_after_np)
        return result

    parameter_after = _parameter_snapshot(surface, optimizer, trainable_params)
    metadata["parameter_after"] = parameter_after
    parameters_changed = _parameters_changed(parameter_before, parameter_after)
    differentiable = bool(requires_grad_true and gradient_norm is not None and gradient_norm > 0 and parameters_changed)
    metadata["per_parameter_grad_norm"] = _per_parameter_grad_norm(surface, optimizer, trainable_params)

    result = SurfaceOptimizationProbeResult(
        probe_id=spec.probe_id,
        status="succeeded" if differentiable else "unsupported",
        surface_class=spec.surface_class,
        objective=spec.objective,
        module_path=module_path,
        can_instantiate=True,
        has_get_optimizer_params=callable(getattr(surface, "get_optimizer_params", None)),
        has_get_optimizer=callable(getattr(surface, "get_optimizer", None)),
        trainable_params=trainable_params,
        differentiable=differentiable,
        autograd_graph_exists=bool(gradient_norm is not None and gradient_norm > 0),
        loss_before=loss_before_val,
        loss_after=loss_after_val,
        parameter_norm_before=_snapshot_norm(parameter_before),
        parameter_norm_after=_snapshot_norm(parameter_after),
        gradient_norm=gradient_norm,
        parameters_changed=parameters_changed,
        optimizer_class=type(optimizer).__name__,
        error_code=None if differentiable else "SURFACE_NOT_DIFFERENTIABLE",
        caveats=caveats if differentiable else [*caveats, "requires_grad/backward/step did not change parameters"],
        metadata=metadata,
    )
    if spec.save_artifacts:
        result.artifact_paths = _save_artifacts(spec, result, loss_trace, phase_before_np, phase_after_np)
    return result


def _import_surface_class(surface_class: str) -> tuple[Any | None, str | None, str | None]:
    repo_path = os.getenv("DEEPLENS_REPO_PATH")
    if repo_path and Path(repo_path).is_dir() and repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    module_path = SURFACE_MODULES.get(surface_class)
    if module_path is None:
        return None, None, f"Unknown surface class: {surface_class}"
    try:
        module = importlib.import_module(module_path)
        return getattr(module, surface_class), module_path, None
    except Exception as exc:
        return None, module_path, str(exc)


def _instantiate_surface(surface_class: str, surface_cls: type, device: str) -> Any:
    if surface_class == "Fresnel":
        return surface_cls(d=0.0, f0=50.0, res=48, device=device)
    if surface_class == "Binary2Phase":
        return surface_cls(
            r=5.0,
            d=0.0,
            order2=1.0,
            order4=0.2,
            order6=0.05,
            order8=0.0,
            order10=0.0,
            order12=0.0,
            device=device,
        )
    if surface_class.endswith("Phase"):
        try:
            return surface_cls(r=5.0, d=0.0, device=device)
        except TypeError:
            return surface_cls(r=5.0, d=0.0)
    try:
        return surface_cls(d=0.0, res=48, device=device)
    except TypeError:
        return surface_cls(d=0.0, res=48)


def _get_optimizer(surface: Any, surface_class: str, learning_rate: float) -> Any:
    import torch

    if callable(getattr(surface, "get_optimizer", None)):
        attempts = [
            ((), {"lr": learning_rate}),
            ((learning_rate,), {}),
            ((), {"lrs": [learning_rate, learning_rate]}),
            (([learning_rate, learning_rate],), {}),
            ((), {}),
        ]
        for args, kwargs in attempts:
            try:
                optimizer = surface.get_optimizer(*args, **kwargs)
                if optimizer is not None:
                    return optimizer
            except TypeError:
                continue
            except Exception:
                continue

    if not callable(getattr(surface, "get_optimizer_params", None)):
        raise RuntimeError(f"{surface_class} has no get_optimizer/get_optimizer_params")
    attempts = [
        ((), {"lr": learning_rate}),
        ((learning_rate,), {}),
        ((), {"lrs": [learning_rate, learning_rate], "optim_mat": False}),
        ((), {"lrs": [learning_rate, learning_rate]}),
        (([learning_rate, learning_rate],), {}),
        ((), {}),
    ]
    errors: list[str] = []
    for args, kwargs in attempts:
        try:
            params = surface.get_optimizer_params(*args, **kwargs)
            return torch.optim.Adam(params)
        except TypeError as exc:
            errors.append(str(exc))
            continue
    raise RuntimeError("; ".join(errors) or "get_optimizer_params failed")


def _surface_phase(surface: Any, surface_class: str, device: str) -> Any:
    import torch

    if callable(getattr(surface, "phase_func", None)):
        return surface.phase_func()
    if callable(getattr(surface, "phi", None)):
        x, y = _xy_grid(device=device)
        return surface.phi(x, y)
    raise RuntimeError(f"{surface_class} does not expose phase_func() or phi(x, y)")


def _xy_grid(size: int = 48, device: str = "cpu") -> tuple[Any, Any]:
    import torch

    coords = torch.linspace(-1.0, 1.0, size, device=device)
    y, x = torch.meshgrid(coords, coords, indexing="ij")
    return x, y


def _phase_loss(phase: Any, objective: str, target_phase: Any = None) -> tuple[Any, Any]:
    import torch

    if objective == "match_target_phase":
        if target_phase is None:
            x, y = _xy_grid(size=phase.shape[-1], device=str(phase.device))
            target_phase = 0.2 * torch.sin(torch.pi * x) + 0.1 * torch.cos(torch.pi * y)
        return torch.nn.functional.mse_loss(phase, target_phase), target_phase
    return phase.pow(2).mean(), target_phase


def _trainable_parameter_names(surface: Any, optimizer: Any, surface_class: str) -> list[str]:
    names = []
    known = SURFACE_TRAINABLE_NAMES.get(surface_class, [])
    params = _optimizer_params(optimizer)
    id_to_known = {id(getattr(surface, name)): name for name in known if hasattr(surface, name)}
    for index, param in enumerate(params):
        names.append(id_to_known.get(id(param), f"param_{index}"))
    return _unique(names)


def _parameter_snapshot(surface: Any, optimizer: Any, names: list[str]) -> dict[str, Any]:
    params = _optimizer_params(optimizer)
    snapshot: dict[str, Any] = {}
    for name, param in zip(names, params):
        snapshot[name] = _tensor_value(param)
    return snapshot


def _per_parameter_grad_norm(surface: Any, optimizer: Any, names: list[str]) -> dict[str, float | None]:
    del surface
    norms: dict[str, float | None] = {}
    for name, param in zip(names, _optimizer_params(optimizer)):
        grad = getattr(param, "grad", None)
        norms[name] = float(grad.detach().norm().cpu().item()) if grad is not None else None
    return norms


def _optimizer_params(optimizer: Any) -> list[Any]:
    params: list[Any] = []
    for group in optimizer.param_groups:
        group_params = group.get("params", [])
        if hasattr(group_params, "detach"):
            params.append(group_params)
        else:
            params.extend(group_params)
    return params


def _requires_grad_true(optimizer: Any) -> bool:
    params = _optimizer_params(optimizer)
    return bool(params) and all(bool(getattr(param, "requires_grad", False)) for param in params)


def _gradient_norm(optimizer: Any) -> float:
    total = 0.0
    for param in _optimizer_params(optimizer):
        grad = getattr(param, "grad", None)
        if grad is not None:
            total += float(grad.detach().norm().cpu().item() ** 2)
    return float(total ** 0.5)


def _tensor_value(value: Any) -> Any:
    tensor = value.detach().cpu()
    if tensor.numel() == 1:
        return float(tensor.item())
    return {"shape": list(tensor.shape), "norm": float(tensor.norm().item())}


def _parameters_changed(before: dict[str, Any], after: dict[str, Any]) -> bool:
    for key, before_value in before.items():
        after_value = after.get(key)
        if isinstance(before_value, dict) or isinstance(after_value, dict):
            before_norm = float((before_value or {}).get("norm", 0.0))
            after_norm = float((after_value or {}).get("norm", 0.0))
            if abs(after_norm - before_norm) > 1e-12:
                return True
        elif after_value is not None and abs(float(after_value) - float(before_value)) > 1e-12:
            return True
    return False


def _snapshot_norm(snapshot: dict[str, Any]) -> float:
    total = 0.0
    for value in snapshot.values():
        if isinstance(value, dict):
            total += float(value.get("norm", 0.0)) ** 2
        else:
            total += float(value) ** 2
    return float(total ** 0.5)


def _to_numpy(tensor: Any) -> np.ndarray:
    return np.asarray(tensor.detach().cpu().numpy())


def _save_artifacts(
    spec: SurfaceOptimizationProbeSpec,
    result: SurfaceOptimizationProbeResult,
    loss_trace: list[dict[str, Any]],
    phase_before: np.ndarray | None,
    phase_after: np.ndarray | None,
) -> list[str]:
    out_dir = Path("workspace/native_optimization") / f"surface_probe_{spec.probe_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "probe_spec.json").write_text(
        json.dumps(spec.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "loss_trace.json").write_text(
        json.dumps(loss_trace, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    (out_dir / "parameter_before.json").write_text(
        json.dumps(result.metadata.get("parameter_before", {}), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "parameter_after.json").write_text(
        json.dumps(result.metadata.get("parameter_after", {}), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if phase_before is not None:
        np.savez_compressed(out_dir / "phase_before.npz", phase=phase_before)
    if phase_after is not None:
        np.savez_compressed(out_dir / "phase_after.npz", phase=phase_after)
    (out_dir / "report.md").write_text("\n".join(_markdown_report(result)), encoding="utf-8")
    result_payload = result.model_dump(mode="json")
    result_payload["artifact_paths"] = [
        str((out_dir / name).relative_to(Path("workspace")))
        for name in [
            "probe_spec.json",
            "probe_result.json",
            "loss_trace.json",
            "parameter_before.json",
            "parameter_after.json",
            "phase_before.npz",
            "phase_after.npz",
            "report.md",
        ]
        if (out_dir / name).exists() or name == "probe_result.json"
    ]
    (out_dir / "probe_result.json").write_text(
        json.dumps(result_payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return result_payload["artifact_paths"]


def _markdown_report(result: SurfaceOptimizationProbeResult) -> list[str]:
    return [
        f"# Surface Optimization Probe: {result.surface_class}",
        "",
        f"- status: {result.status}",
        f"- objective: {result.objective}",
        f"- differentiable: {result.differentiable}",
        f"- gradient_norm: {result.gradient_norm}",
        f"- parameters_changed: {result.parameters_changed}",
        f"- loss_before: {result.loss_before}",
        f"- loss_after: {result.loss_after}",
    ]


def _unsupported(
    spec: SurfaceOptimizationProbeSpec,
    error_code: str,
    error_message: str,
    metadata: dict[str, Any],
    caveats: list[str],
    module_path: str | None = None,
) -> SurfaceOptimizationProbeResult:
    return SurfaceOptimizationProbeResult(
        probe_id=spec.probe_id,
        status="unsupported",
        surface_class=spec.surface_class,
        objective=spec.objective,
        module_path=module_path,
        error_code=error_code,
        error_message=error_message,
        caveats=[*caveats, error_message],
        metadata=metadata,
    )


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result
