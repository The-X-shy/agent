"""Minimal native differentiable optimization probe runner.

Executes the full autograd chain:
    optical parameter -> PSF simulation -> scalar loss -> backward ->
    optimizer.step -> parameter change

And verifies that gradient flows and parameters update. This is the minimal
smoke test for DeepLens native differentiable optical optimization.

IMPORTANT: This probe must NOT silently fallback to mock, must NOT fake
parameter updates, and must NOT claim native optimization unless it actually
executes the full backward + optimizer.step chain.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from optiresearch.schemas.native_optimization import (
    NativeOptimizationProbeResult,
    NativeOptimizationProbeSpec,
)

MODULE_MAP = {
    "ParaxialLens": "deeplens.paraxiallens",
    "GeoLens": "deeplens.geolens",
    "DiffractiveLens": "deeplens.diffraclens",
    "HybridLens": "deeplens.hybridlens",
    "PSFNetLens": "deeplens.psfnetlens",
}


def run_native_optimization_probe(spec: NativeOptimizationProbeSpec) -> NativeOptimizationProbeResult:
    """Run a native optimization probe and return a structured result.

    Executes the minimal autograd chain: instantiate lens -> generate PSF ->
    compute loss -> backward -> optimizer.step -> check parameter changes.
    """

    caveats: list[str] = []
    metadata: dict[str, Any] = {"spec_device": spec.device, "strict_native": spec.strict_native}

    # Step 1: Import lens class
    try:
        lens_cls, import_path, import_error = _import_lens_class(spec.lens_class)
    except Exception as exc:
        return NativeOptimizationProbeResult(
            probe_id=spec.probe_id,
            status="unsupported",
            lens_class=spec.lens_class,
            objective=spec.objective,
            realization_level="unavailable",
            error_code="IMPORT_FAILED",
            error_message=str(exc),
            caveats=[f"Failed to import {spec.lens_class}: {exc}"],
            metadata=metadata,
        )

    if lens_cls is None:
        return NativeOptimizationProbeResult(
            probe_id=spec.probe_id,
            status="unsupported",
            lens_class=spec.lens_class,
            objective=spec.objective,
            realization_level="unavailable",
            error_code="DEEPLENS_NOT_INSTALLED",
            error_message=import_error or "DeepLens not installed",
            caveats=caveats,
            metadata=metadata,
        )

    # Step 2: Instantiate lens
    try:
        lens_instance, instantiation_caveats = _instantiate_lens(spec.lens_class, lens_cls, spec.device)
        caveats.extend(instantiation_caveats)
    except Exception as exc:
        return NativeOptimizationProbeResult(
            probe_id=spec.probe_id,
            status="unsupported",
            lens_class=spec.lens_class,
            objective=spec.objective,
            realization_level="semi_native",
            error_code="INSTANTIATION_FAILED",
            error_message=str(exc),
            caveats=[f"Failed to instantiate {spec.lens_class}: {exc}"],
            metadata=metadata,
        )

    if lens_instance is None:
        return NativeOptimizationProbeResult(
            probe_id=spec.probe_id,
            status="unsupported",
            lens_class=spec.lens_class,
            objective=spec.objective,
            realization_level="unavailable",
            error_code="LENS_NOT_INSTANTIABLE",
            error_message=f"Cannot instantiate {spec.lens_class} — requires lens file but none found",
            caveats=caveats,
            metadata=metadata,
        )

    # Step 3: Activate gradients
    try:
        grad_ok, grad_error, grad_caveats = _activate_grad(lens_instance)
        caveats.extend(grad_caveats)
        if not grad_ok and spec.strict_native:
            return NativeOptimizationProbeResult(
                probe_id=spec.probe_id,
                status="unsupported",
                lens_class=spec.lens_class,
                objective=spec.objective,
                realization_level="semi_native",
                error_code="GRAD_ACTIVATION_FAILED",
                error_message=grad_error or "activate_grad failed",
                caveats=caveats,
                metadata=metadata,
            )
    except Exception as exc:
        caveats.append(f"activate_grad exception: {exc}")
        if spec.strict_native:
            return NativeOptimizationProbeResult(
                probe_id=spec.probe_id,
                status="unsupported",
                lens_class=spec.lens_class,
                objective=spec.objective,
                realization_level="semi_native",
                error_code="GRAD_ACTIVATION_FAILED",
                error_message=str(exc),
                caveats=caveats,
                metadata=metadata,
            )

    # Step 4: Generate initial PSF (keep gradients)
    try:
        psf_raw = _generate_psf(lens_instance, spec.lens_class, detach=False)
    except Exception as exc:
        return NativeOptimizationProbeResult(
            probe_id=spec.probe_id,
            status="failed",
            lens_class=spec.lens_class,
            objective=spec.objective,
            realization_level="semi_native",
            error_code="PSF_GENERATION_FAILED",
            error_message=str(exc),
            caveats=[f"Failed to generate PSF: {exc}", *caveats],
            metadata=metadata,
        )

    # Save detached numpy copy for artifacts
    psf_before_np = np.asarray(psf_raw.detach().cpu().numpy()) if hasattr(psf_raw, "detach") else np.asarray(psf_raw)

    # Step 5: Compute loss
    try:
        import torch
        psf_tensor = torch.as_tensor(psf_raw, dtype=torch.float32, device=spec.device)
        loss_before, target_psf = _compute_loss(psf_tensor, spec.objective)
        loss_before_val = float(loss_before.detach().cpu().item())
    except Exception as exc:
        return NativeOptimizationProbeResult(
            probe_id=spec.probe_id,
            status="failed",
            lens_class=spec.lens_class,
            objective=spec.objective,
            realization_level="semi_native",
            error_code="LOSS_COMPUTATION_FAILED",
            error_message=str(exc),
            caveats=[f"Failed to compute loss: {exc}", *caveats],
            metadata=metadata,
        )

    # Step 6: Record parameter norm before
    param_norm_before = _compute_parameter_norm(lens_instance)

    # Step 7: backward
    try:
        loss_before.backward()
        gradient_norm = _compute_gradient_norm(lens_instance)
        autograd_graph_exists = gradient_norm is not None and gradient_norm > 0
    except Exception as exc:
        gradient_norm = None
        autograd_graph_exists = False
        caveats.append(f"backward failed: {exc}")

    # Step 8: Get optimizer and step
    try:
        optimizer, opt_class_name = _get_optimizer(lens_instance, spec.learning_rate)
        if optimizer is not None:
            optimizer.step()
            optimizer_ok = True
        else:
            optimizer_ok = False
            caveats.append("No optimizer available")
    except Exception as exc:
        optimizer_ok = False
        opt_class_name = None
        caveats.append(f"optimizer.step failed: {exc}")

    # Step 9: Check parameter change
    param_norm_after = _compute_parameter_norm(lens_instance)
    parameters_changed = (
        param_norm_before is not None
        and param_norm_after is not None
        and abs(param_norm_after - param_norm_before) > 1e-10
    )

    # Step 10: Compute loss after
    try:
        psf_after_raw = _generate_psf(lens_instance, spec.lens_class, detach=False)
        psf_after_tensor = torch.as_tensor(psf_after_raw, dtype=torch.float32, device=spec.device)
        loss_after_tensor, _ = _compute_loss(psf_after_tensor, spec.objective, target_psf=target_psf)
        loss_after_val = float(loss_after_tensor.detach().cpu().item())
        psf_after_np = np.asarray(psf_after_raw.detach().cpu().numpy()) if hasattr(psf_after_raw, "detach") else np.asarray(psf_after_raw)
    except Exception as exc:
        loss_after_val = None
        psf_after_np = None
        caveats.append(f"loss_after computation failed: {exc}")

    # Step 11: Determine realization level
    differentiable = autograd_graph_exists and parameters_changed
    native_parameter_update = differentiable

    if differentiable:
        realization_level = "native"
        status = "succeeded"
    elif autograd_graph_exists and not parameters_changed:
        realization_level = "semi_native"
        status = "failed"
        if not spec.strict_native:
            status = "unsupported"
            caveats.append("Gradient exists but parameters did not change")
    elif autograd_graph_exists is False:
        realization_level = "semi_native"
        status = "unsupported" if spec.strict_native else "failed"
        caveats.append("No autograd graph detected")
    else:
        realization_level = "semi_native"
        status = "unsupported"

    # Step 12: Save artifacts
    artifact_paths: list[str] = []
    if spec.save_artifacts:
        try:
            artifact_paths = _save_probe_artifacts(
                spec, psf_before_np, psf_after_np, loss_before_val, loss_after_val,
                param_norm_before, param_norm_after, gradient_norm,
                parameters_changed, caveats,
            )
        except Exception as exc:
            caveats.append(f"Artifact saving failed: {exc}")

    return NativeOptimizationProbeResult(
        probe_id=spec.probe_id,
        status=status,
        lens_class=spec.lens_class,
        objective=spec.objective,
        realization_level=realization_level,
        differentiable=differentiable,
        native_parameter_update=native_parameter_update,
        autograd_graph_exists=autograd_graph_exists,
        loss_before=loss_before_val,
        loss_after=loss_after_val,
        parameter_norm_before=param_norm_before,
        parameter_norm_after=param_norm_after,
        gradient_norm=gradient_norm,
        parameters_changed=parameters_changed,
        optimizer_class=opt_class_name,
        artifact_paths=artifact_paths,
        error_code=None if status == "succeeded" else _error_code_for_status(status),
        caveats=caveats,
        metadata=metadata,
    )


def _import_lens_class(cls_name: str) -> tuple[Any, str | None, str | None]:
    """Import a lens class from DeepLens. Returns (cls, import_path, error)."""
    module_path = MODULE_MAP.get(cls_name)
    if module_path is None:
        return None, None, f"Unknown lens class: {cls_name}"

    repo_path = os.getenv("DEEPLENS_REPO_PATH", "")
    if repo_path and Path(repo_path).is_dir() and repo_path not in sys.path:
        sys.path.insert(0, repo_path)

    try:
        mod = importlib.import_module(module_path)
        cls = getattr(mod, cls_name, None)
        if cls is None:
            return None, module_path, f"Class {cls_name} not found in {module_path}"
        return cls, module_path, None
    except ImportError as exc:
        return None, module_path, str(exc)
    except Exception as exc:
        return None, module_path, str(exc)


def _instantiate_lens(cls_name: str, cls: type, device: str) -> tuple[Any, list[str]]:
    """Try to instantiate a lens with minimal arguments."""
    caveats: list[str] = []

    if cls_name == "ParaxialLens":
        try:
            instance = cls(foclen=50.0, fnum=2.8, device=device)
            return instance, caveats
        except Exception as exc:
            caveats.append(f"ParaxialLens(foclen=50, fnum=2.8) failed: {exc}")
        try:
            instance = cls(device=device)
            return instance, caveats
        except Exception as exc:
            caveats.append(f"ParaxialLens(device={device}) failed: {exc}")
        try:
            instance = cls()
            return instance, caveats
        except Exception as exc:
            caveats.append(f"ParaxialLens() failed: {exc}")
        return None, caveats

    # For file-based lens classes
    lens_file = _find_lens_file(cls_name)
    if lens_file is not None:
        try:
            instance = cls(str(lens_file), device=device)
            caveats.append(f"Using lens file: {lens_file}")
            return instance, caveats
        except Exception as exc:
            caveats.append(f"{cls_name}(file, device={device}) failed: {exc}")
        try:
            instance = cls(str(lens_file))
            caveats.append(f"Using lens file: {lens_file} (no device arg)")
            return instance, caveats
        except Exception as exc:
            caveats.append(f"{cls_name}(file) failed: {exc}")

    # Try no-arg
    try:
        instance = cls()
        caveats.append(f"{cls_name} instantiated with no args")
        return instance, caveats
    except Exception:
        pass

    return None, caveats


def _find_lens_file(cls_name: str) -> Path | None:
    """Search for a suitable lens JSON file."""
    repo_path = os.getenv("DEEPLENS_REPO_PATH", "")
    if not repo_path:
        return None

    repo = Path(repo_path)
    search_dirs = [
        repo / "deeplens" / "samples",
        repo / "samples",
        repo / "examples",
        repo / "lenses",
        repo / "deeplens" / "lenses",
    ]

    for search_dir in search_dirs:
        if not search_dir.is_dir():
            continue
        for candidate in sorted(search_dir.rglob("*.json")):
            return candidate
        for candidate in sorted(search_dir.rglob("*.JSON")):
            return candidate

    return None


def _activate_grad(lens_instance: Any) -> tuple[bool, str | None, list[str]]:
    """Try to activate gradient tracking. Returns (ok, error, caveats)."""
    caveats: list[str] = []
    if not hasattr(lens_instance, "activate_grad"):
        return False, "No activate_grad method", caveats
    try:
        result = lens_instance.activate_grad(True)
        return True, None, caveats
    except NotImplementedError:
        return False, "activate_grad raises NotImplementedError (not overridden by subclass)", caveats
    except Exception as exc:
        return False, str(exc), caveats


def _generate_psf(lens_instance: Any, cls_name: str, detach: bool = True) -> Any:
    """Generate PSF from lens instance.

    Args:
        lens_instance: The lens to generate PSF from.
        cls_name: Name of the lens class (for error messages).
        detach: If True, detach and convert to numpy. If False, keep as raw
                tensor so autograd can track gradients.

    Returns:
        numpy array (if detach=True) or raw tensor (if detach=False).
    """
    errors: list[str] = []

    # Try psf() with various argument patterns
    if hasattr(lens_instance, "psf"):
        # For DeepLens ParaxialLens, psf expects points=[N,3] tensor
        import torch
        for args, kwargs in [
            ([], {}),
            ([], {"points": torch.zeros(1, 3), "ks": 32}),
            ([], {"points": torch.tensor([[0.0, 0.0, 0.0]]), "ks": 32}),
            ([], {"points": [0.55], "ks": 32}),
            ([], {"points": [0.0], "ks": 32}),
            ([torch.zeros(1, 3)], {"ks": 32}),
            ([torch.zeros(1, 3)], {}),
        ]:
            try:
                result = lens_instance.psf(*args, **kwargs)
                if detach and hasattr(result, "detach"):
                    return np.asarray(result.detach().cpu().numpy())
                return result
            except Exception as exc:
                errors.append(f"psf({args}, {kwargs}): {exc}")
                continue

    # Try render()
    if hasattr(lens_instance, "render"):
        for args, kwargs in [
            ([], {}),
            ([], {"points": [0.55]}),
        ]:
            try:
                result = lens_instance.render(*args, **kwargs)
                if detach and hasattr(result, "detach"):
                    return np.asarray(result.detach().cpu().numpy())
                return result
            except Exception as exc:
                errors.append(f"render({args}, {kwargs}): {exc}")
                continue

    # Try forward()
    if hasattr(lens_instance, "forward"):
        try:
            result = lens_instance.forward()
            if detach and hasattr(result, "detach"):
                return np.asarray(result.detach().cpu().numpy())
            return result
        except Exception as exc:
            errors.append(f"forward(): {exc}")

    raise RuntimeError(f"No PSF method found or all call patterns failed on {cls_name}: {'; '.join(errors)}")


def _compute_loss(
    psf: Any,
    objective: str,
    target_psf: Any = None,
) -> tuple[Any, Any]:
    """Compute scalar loss from PSF tensor. Returns (loss, target_psf)."""
    import torch

    if objective == "maximize_center_intensity":
        loss = -psf.max()
        return loss, target_psf

    if objective == "match_target_psf":
        if target_psf is None:
            # Generate a simple Gaussian target
            H, W = psf.shape[-2], psf.shape[-1]
            y, x = torch.meshgrid(
                torch.linspace(-1, 1, H, device=psf.device),
                torch.linspace(-1, 1, W, device=psf.device),
                indexing="ij",
            )
            target_psf = torch.exp(-(x**2 + y**2) / 0.1)
            target_psf = target_psf / target_psf.sum()
        loss = torch.nn.functional.mse_loss(psf, target_psf)
        return loss, target_psf

    # minimize_psf_width (default): spatial variance
    H, W = psf.shape[-2], psf.shape[-1]
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=psf.device),
        torch.linspace(-1, 1, W, device=psf.device),
        indexing="ij",
    )
    total = psf.sum() + 1e-8
    cx = (psf * x).sum() / total
    cy = (psf * y).sum() / total
    var_x = ((x - cx) ** 2 * psf).sum() / total
    var_y = ((y - cy) ** 2 * psf).sum() / total
    loss = var_x + var_y
    return loss, target_psf


def _compute_parameter_norm(lens_instance: Any) -> float | None:
    """Compute total L2 norm of all trainable parameters."""
    if not hasattr(lens_instance, "parameters"):
        return None
    try:
        total = 0.0
        for p in lens_instance.parameters():
            if hasattr(p, "data"):
                total += float(p.data.norm().item()) if hasattr(p.data, "norm") else 0.0
        return total
    except Exception:
        return None


def _compute_gradient_norm(lens_instance: Any) -> float | None:
    """Compute total gradient norm across all parameters."""
    if not hasattr(lens_instance, "parameters"):
        return None
    try:
        total_norm = 0.0
        for p in lens_instance.parameters():
            if hasattr(p, "grad") and p.grad is not None:
                total_norm += float(p.grad.norm().item() ** 2)
        return float(total_norm ** 0.5)
    except Exception:
        return None


def _get_optimizer(lens_instance: Any, learning_rate: float) -> tuple[Any, str | None]:
    """Get optimizer. Prefers lens.get_optimizer(), falls back to Adam."""
    if hasattr(lens_instance, "get_optimizer"):
        try:
            opt = lens_instance.get_optimizer()
            if opt is not None:
                cls_name = type(opt).__name__
                return opt, cls_name
        except Exception:
            pass

    # Fallback: manual Adam
    try:
        import torch
        if hasattr(lens_instance, "parameters"):
            params = [p for p in lens_instance.parameters() if hasattr(p, "requires_grad") and p.requires_grad]
            if params:
                opt = torch.optim.Adam(params, lr=learning_rate)
                return opt, "Adam (manual fallback)"
    except Exception:
        pass

    return None, None


def _error_code_for_status(status: str) -> str | None:
    if status == "unsupported":
        return "NATIVE_OPTIMIZATION_NOT_SUPPORTED"
    return "PROBE_FAILED"


def _save_probe_artifacts(
    spec: NativeOptimizationProbeSpec,
    psf_before: np.ndarray,
    psf_after: np.ndarray,
    loss_before: float | None,
    loss_after: float | None,
    param_norm_before: float | None,
    param_norm_after: float | None,
    gradient_norm: float | None,
    parameters_changed: bool | None,
    caveats: list[str],
) -> list[str]:
    """Save probe artifacts. Returns list of relative paths."""
    output_dir = Path("workspace/native_optimization") / spec.probe_id
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    # Probe spec
    spec_path = output_dir / "probe_spec.json"
    spec_path.write_text(
        json.dumps(spec.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    paths.append(str(spec_path.relative_to(output_dir.parent.parent)))

    # PSF before/after
    npz_path = output_dir / "psf_before.npz"
    np.savez_compressed(npz_path, psf=psf_before)
    paths.append(str(npz_path.relative_to(output_dir.parent.parent)))

    if psf_after is not None:
        npz_path = output_dir / "psf_after.npz"
        np.savez_compressed(npz_path, psf=psf_after)
        paths.append(str(npz_path.relative_to(output_dir.parent.parent)))

    # Loss trace
    loss_trace = {
        "loss_before": loss_before,
        "loss_after": loss_after,
    }
    trace_path = output_dir / "loss_trace.json"
    trace_path.write_text(
        json.dumps(loss_trace, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    paths.append(str(trace_path.relative_to(output_dir.parent.parent)))

    # Parameter snapshots
    param_snap = {
        "parameter_norm_before": param_norm_before,
        "parameter_norm_after": param_norm_after,
        "gradient_norm": gradient_norm,
        "parameters_changed": parameters_changed,
    }
    snap_path = output_dir / "parameter_snapshot.json"
    snap_path.write_text(
        json.dumps(param_snap, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    paths.append(str(snap_path.relative_to(output_dir.parent.parent)))

    # Markdown report
    md_lines = _build_probe_markdown(
        spec, loss_before, loss_after, param_norm_before, param_norm_after,
        gradient_norm, parameters_changed, caveats,
    )
    md_path = output_dir / "native_probe_report.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    paths.append(str(md_path.relative_to(output_dir.parent.parent)))

    return paths


def _build_probe_markdown(
    spec: NativeOptimizationProbeSpec,
    loss_before: float | None,
    loss_after: float | None,
    param_norm_before: float | None,
    param_norm_after: float | None,
    gradient_norm: float | None,
    parameters_changed: bool | None,
    caveats: list[str],
) -> list[str]:
    lines = [
        f"# Native Optimization Probe: {spec.probe_id}",
        "",
        f"**Lens class:** `{spec.lens_class}`",
        f"**Objective:** `{spec.objective}`",
        f"**Device:** `{spec.device}`",
        f"**Max steps:** {spec.max_steps}",
        f"**Learning rate:** {spec.learning_rate}",
        "",
        "## Results",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| loss_before | {loss_before} |",
        f"| loss_after | {loss_after} |",
        f"| parameter_norm_before | {param_norm_before} |",
        f"| parameter_norm_after | {param_norm_after} |",
        f"| gradient_norm | {gradient_norm} |",
        f"| parameters_changed | {parameters_changed} |",
        "",
    ]
    if caveats:
        lines.append("## Caveats")
        lines.append("")
        for c in caveats:
            lines.append(f"- {c}")
        lines.append("")
    return lines
