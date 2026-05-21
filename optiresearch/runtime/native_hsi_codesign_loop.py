"""Native Optical-HSI CoDesign optimization loop for Phase 20.

Connects DeepLens native trainable optical parameters to HSI proxy loss
through a differentiable phase-to-PSF bridge.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from optiresearch.adapters.deeplens_differentiable_bridge import (
    Binary2PhaseHSIBridge,
    FresnelHSIBridge,
)
from optiresearch.hsi.differentiable_proxy import (
    generate_torch_synthetic_hsi,
    hsi_proxy_loss,
    make_measurement_from_psf_torch,
    reconstruct_proxy_torch,
)
from optiresearch.schemas.native_hsi_codesign import (
    NativeOpticalHSICoDesignResult,
    NativeOpticalHSICoDesignSpec,
)

BRIDGE_CLASSES = {
    "Fresnel": FresnelHSIBridge,
    "Binary2Phase": Binary2PhaseHSIBridge,
}


def run_native_optical_hsi_codesign(
    spec: NativeOpticalHSICoDesignSpec,
) -> NativeOpticalHSICoDesignResult:
    """Run native optical-HSI co-design optimization loop."""
    metadata: dict[str, Any] = {"optimizer_step_executed": False}
    caveats: list[str] = []

    bridge_cls = BRIDGE_CLASSES.get(spec.optical_component)
    if bridge_cls is None:
        return _unsupported(spec, "UNSUPPORTED_COMPONENT",
                            f"No bridge for {spec.optical_component}", metadata, caveats)

    try:
        bridge = bridge_cls(device=spec.device)
        bridge.build_component()
    except Exception as exc:
        return _unsupported(spec, "BUILD_FAILED", str(exc), metadata, caveats)

    try:
        optimizer = bridge.get_optimizer(learning_rate=spec.learning_rate)
    except Exception as exc:
        return _unsupported(spec, "OPTIMIZER_UNAVAILABLE", str(exc), metadata, caveats)

    parameter_before = bridge.parameter_snapshot()
    metadata["parameter_before"] = parameter_before

    gradient_norm: float | None = None
    hsi_loss_before: float | None = None
    hsi_loss_after: float | None = None
    psf_before_np: np.ndarray | None = None
    psf_after_np: np.ndarray | None = None
    loss_trace: list[dict[str, Any]] = []
    autograd_break_detected = False

    try:
        hsi_target = generate_torch_synthetic_hsi(
            batch=1, bands=spec.bands, height=spec.image_size,
            width=spec.image_size, device=spec.device,
        )

        psf = bridge.psf_from_component_torch(num_bands=spec.bands, psf_size=spec.psf_size)
        psf_before_np = psf.detach().cpu().numpy()

        measurement = make_measurement_from_psf_torch(hsi_target, psf)
        recon = reconstruct_proxy_torch(measurement, psf, bands=spec.bands)
        loss = hsi_proxy_loss(recon, hsi_target, mode="mse")
        hsi_loss_before = float(loss.detach().cpu().item())

        if not loss.requires_grad:
            autograd_break_detected = True
            return _autograd_break(spec, "LOSS_NOT_DIFFERENTIABLE",
                                   "HSI proxy loss does not require grad", metadata, caveats,
                                   hsi_loss_before=hsi_loss_before)

        optimizer.zero_grad()
        loss.backward()

        gradient_norm = bridge.gradient_norm()
        if gradient_norm is None or gradient_norm == 0.0:
            autograd_break_detected = True
            return _autograd_break(spec, "ZERO_GRADIENT",
                                   "Gradient norm is zero after backward", metadata, caveats,
                                   hsi_loss_before=hsi_loss_before, gradient_norm=gradient_norm)

        optimizer.step()
        metadata["optimizer_step_executed"] = True

        loss_trace.append({
            "step": 0,
            "loss": hsi_loss_before,
            "gradient_norm": gradient_norm,
        })

        psf_after = bridge.psf_from_component_torch(num_bands=spec.bands, psf_size=spec.psf_size)
        psf_after_np = psf_after.detach().cpu().numpy()
        measurement_after = make_measurement_from_psf_torch(hsi_target, psf_after)
        recon_after = reconstruct_proxy_torch(measurement_after, psf_after, bands=spec.bands)
        loss_after = hsi_proxy_loss(recon_after, hsi_target, mode="mse")
        hsi_loss_after = float(loss_after.detach().cpu().item())

    except Exception as exc:
        return _failed(spec, "LOOP_FAILED", str(exc), metadata, caveats,
                       hsi_loss_before=hsi_loss_before, gradient_norm=gradient_norm)

    parameter_after = bridge.parameter_snapshot()
    metadata["parameter_after"] = parameter_after

    parameters_changed = _parameters_changed(parameter_before, parameter_after)
    differentiable = bool(
        gradient_norm is not None and gradient_norm > 0
        and parameters_changed
        and metadata["optimizer_step_executed"]
    )
    evidence_level = "native_hsi_proxy" if differentiable else None

    result = NativeOpticalHSICoDesignResult(
        run_id=spec.run_id,
        status="succeeded" if differentiable else "unsupported",
        optical_component=spec.optical_component,
        objective=spec.objective,
        differentiable=differentiable,
        native_parameter_update=differentiable,
        hsi_loss_before=hsi_loss_before,
        hsi_loss_after=hsi_loss_after,
        gradient_norm=gradient_norm,
        parameters_changed=parameters_changed,
        optimizer_step_executed=metadata["optimizer_step_executed"],
        autograd_break_detected=autograd_break_detected,
        evidence_level=evidence_level,
        caveats=caveats if differentiable else [*caveats, "HSI loss did not backprop to optical params"],
        metadata=metadata,
    )

    if spec.save_artifacts:
        result.artifact_paths = _save_artifacts(
            spec, result, loss_trace, parameter_before, parameter_after,
            psf_before_np, psf_after_np,
        )
    return result


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


def _save_artifacts(
    spec: NativeOpticalHSICoDesignSpec,
    result: NativeOpticalHSICoDesignResult,
    loss_trace: list[dict[str, Any]],
    parameter_before: dict[str, Any],
    parameter_after: dict[str, Any],
    psf_before: np.ndarray | None,
    psf_after: np.ndarray | None,
) -> list[str]:
    out_dir = Path("workspace/native_hsi_codesign") / spec.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "spec.json").write_text(
        json.dumps(spec.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "result.json").write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    (out_dir / "loss_trace.json").write_text(
        json.dumps(loss_trace, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    (out_dir / "parameter_before.json").write_text(
        json.dumps(parameter_before, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "parameter_after.json").write_text(
        json.dumps(parameter_after, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if psf_before is not None:
        np.savez_compressed(out_dir / "psf_before.npz", psf=psf_before)
    if psf_after is not None:
        np.savez_compressed(out_dir / "psf_after.npz", psf=psf_after)

    metrics = {
        "hsi_loss_before": result.hsi_loss_before,
        "hsi_loss_after": result.hsi_loss_after,
        "gradient_norm": result.gradient_norm,
        "parameters_changed": result.parameters_changed,
    }
    (out_dir / "hsi_proxy_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "report.md").write_text("\n".join(_report_lines(result)), encoding="utf-8")

    paths = []
    for name in [
        "spec.json", "result.json", "loss_trace.json", "parameter_before.json",
        "parameter_after.json", "psf_before.npz", "psf_after.npz",
        "hsi_proxy_metrics.json", "report.md",
    ]:
        if (out_dir / name).exists():
            paths.append(str((out_dir / name).relative_to(Path("workspace"))))
    return paths


def _report_lines(result: NativeOpticalHSICoDesignResult) -> list[str]:
    return [
        f"# Native Optical-HSI CoDesign: {result.optical_component}",
        "",
        f"- status: {result.status}",
        f"- objective: {result.objective}",
        f"- differentiable: {result.differentiable}",
        f"- evidence_level: {result.evidence_level}",
        f"- hsi_loss_before: {result.hsi_loss_before}",
        f"- hsi_loss_after: {result.hsi_loss_after}",
        f"- gradient_norm: {result.gradient_norm}",
        f"- parameters_changed: {result.parameters_changed}",
        f"- optimizer_step_executed: {result.optimizer_step_executed}",
        f"- autograd_break_detected: {result.autograd_break_detected}",
        f"- caveats: {result.caveats}",
    ]


def _unsupported(
    spec: NativeOpticalHSICoDesignSpec,
    error_code: str,
    error_message: str,
    metadata: dict[str, Any],
    caveats: list[str],
) -> NativeOpticalHSICoDesignResult:
    return NativeOpticalHSICoDesignResult(
        run_id=spec.run_id,
        status="unsupported",
        optical_component=spec.optical_component,
        objective=spec.objective,
        error_code=error_code,
        error_message=error_message,
        caveats=[*caveats, error_message],
        metadata=metadata,
    )


def _autograd_break(
    spec: NativeOpticalHSICoDesignSpec,
    error_code: str,
    error_message: str,
    metadata: dict[str, Any],
    caveats: list[str],
    hsi_loss_before: float | None = None,
    gradient_norm: float | None = None,
) -> NativeOpticalHSICoDesignResult:
    return NativeOpticalHSICoDesignResult(
        run_id=spec.run_id,
        status="unsupported",
        optical_component=spec.optical_component,
        objective=spec.objective,
        differentiable=False,
        autograd_break_detected=True,
        hsi_loss_before=hsi_loss_before,
        gradient_norm=gradient_norm,
        error_code=error_code,
        error_message=error_message,
        caveats=[*caveats, error_message],
        metadata=metadata,
    )


def _failed(
    spec: NativeOpticalHSICoDesignSpec,
    error_code: str,
    error_message: str,
    metadata: dict[str, Any],
    caveats: list[str],
    hsi_loss_before: float | None = None,
    gradient_norm: float | None = None,
) -> NativeOpticalHSICoDesignResult:
    return NativeOpticalHSICoDesignResult(
        run_id=spec.run_id,
        status="failed",
        optical_component=spec.optical_component,
        objective=spec.objective,
        differentiable=False,
        hsi_loss_before=hsi_loss_before,
        gradient_norm=gradient_norm,
        error_code=error_code,
        error_message=error_message,
        caveats=[*caveats, error_message],
        metadata=metadata,
    )
