"""DeepLens wave-optics probe runtime for Phase 22.

Probes GeoLens's native differentiable PSF path:
  geometric: surface params → ray tracing → Monte Carlo binning → PSF (differentiable)
  coherent: surface params → ray tracing → pupil field → ASM (NOT differentiable in practice)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from optiresearch.adapters.geolens_waveoptics_bridge import GeoLensWaveOpticsBridge
from optiresearch.schemas.deeplens_waveoptics_probe import (
    DeepLensWaveOpticsProbeResult,
    DeepLensWaveOpticsProbeSpec,
)


def run_deeplens_waveoptics_probe(
    spec: DeepLensWaveOpticsProbeSpec,
) -> DeepLensWaveOpticsProbeResult:
    metadata: dict[str, Any] = {"optimizer_step_executed": False}
    caveats: list[str] = []

    if spec.strict_waveoptics and spec.allow_phase_to_fft_proxy:
        return _unsupported(spec, "INCOMPATIBLE_OPTIONS",
                            "strict_waveoptics and allow_phase_to_fft_proxy conflict", metadata, caveats)

    try:
        bridge = GeoLensWaveOpticsBridge(device=spec.device)
        bridge.build_component(lens_file=spec.lens_file)
    except Exception as exc:
        return _unsupported(spec, "BUILD_FAILED", str(exc), metadata, caveats)

    try:
        optimizer = bridge.get_optimizer(learning_rate=spec.learning_rate)
    except Exception as exc:
        return _unsupported(spec, "OPTIMIZER_UNAVAILABLE", str(exc), metadata, caveats)

    param_before = bridge.parameter_snapshot()
    metadata["optical_parameter_before"] = param_before

    gradient_norm: float | None = None
    loss_before: float | None = None
    loss_after: float | None = None

    try:
        psf = bridge.psf_from_component_torch(wvln=0.55, ks=spec.psf_size)
        psf_requires_grad = bool(getattr(psf, "requires_grad", False))
        if spec.strict_waveoptics and not psf_requires_grad:
            return _unsupported(spec, "PSF_NOT_DIFFERENTIABLE",
                                "PSF does not require grad", metadata, caveats)

        target = torch.zeros_like(psf)
        center = spec.psf_size // 2
        target[center, center] = 1.0
        loss = torch.nn.functional.mse_loss(psf, target)
        loss_before = float(loss.detach().cpu().item())

        if not loss.requires_grad:
            return _unsupported(spec, "LOSS_NOT_DIFFERENTIABLE",
                                "Loss does not require grad", metadata, caveats)

        optimizer.zero_grad()
        loss.backward()
        gradient_norm = bridge.gradient_norm()

        if gradient_norm == 0.0:
            return _unsupported(spec, "ZERO_GRADIENT",
                                "Gradient norm is zero", metadata, caveats)

        optimizer.step()
        metadata["optimizer_step_executed"] = True

        psf_after = bridge.psf_from_component_torch(wvln=0.55, ks=spec.psf_size)
        loss_after_val = torch.nn.functional.mse_loss(psf_after, target)
        loss_after = float(loss_after_val.detach().cpu().item())

    except Exception as exc:
        return _failed(spec, "PROBE_FAILED", str(exc), metadata, caveats,
                       loss_before=loss_before, gradient_norm=gradient_norm)

    import torch
    param_after = bridge.parameter_snapshot()
    metadata["optical_parameter_after"] = param_after
    params_changed = _params_changed(param_before, param_after)

    differentiable = bool(gradient_norm and gradient_norm > 0 and params_changed and metadata["optimizer_step_executed"])
    evidence_level = "native_full_waveoptics" if differentiable else None

    result = DeepLensWaveOpticsProbeResult(
        run_id=spec.run_id,
        status="succeeded" if differentiable else "unsupported",
        candidate=spec.candidate,
        lens_file=spec.lens_file,
        full_wave_optics=True,
        phase_to_fft_proxy_used=False,
        differentiable=differentiable,
        native_parameter_update=differentiable,
        loss_before=loss_before,
        loss_after=loss_after,
        optical_gradient_norm=gradient_norm,
        optical_parameter_before=param_before,
        optical_parameter_after=param_after,
        optical_parameters_changed=params_changed,
        psf_requires_grad=psf_requires_grad,
        autograd_graph_exists=differentiable,
        deeplens_native_wave_path="geolens.psf_pupil_prop",
        evidence_level=evidence_level,
        optimizer_step_executed=metadata["optimizer_step_executed"],
        caveats=caveats if differentiable else [*caveats, "Wave-optics PSF did not produce differentiable parameter update"],
        metadata=metadata,
    )

    if spec.save_artifacts:
        result.artifact_paths = _save_artifacts(spec, result, param_before, param_after)
    return result


def _params_changed(before: dict, after: dict) -> bool:
    for key, bv in before.items():
        av = after.get(key)
        if isinstance(bv, dict):
            bn = float((bv or {}).get("norm", 0))
            an = float((av or {}).get("norm", 0))
            if abs(an - bn) > 1e-12:
                return True
        elif av is not None and abs(float(av) - float(bv)) > 1e-12:
            return True
    return False


def _save_artifacts(spec, result, before, after):
    out_dir = Path("workspace/waveoptics_probe") / spec.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "spec.json").write_text(json.dumps(spec.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "result.json").write_text(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (out_dir / "report.md").write_text(f"# Wave-Optics Probe: {spec.candidate}\n\n- status: {result.status}\n- differentiable: {result.differentiable}\n- full_wave_optics: {result.full_wave_optics}\n- evidence_level: {result.evidence_level}\n- gradient_norm: {result.optical_gradient_norm}\n- parameters_changed: {result.optical_parameters_changed}\n", encoding="utf-8")
    return [str((out_dir / n).relative_to(Path("workspace"))) for n in ["spec.json", "result.json", "report.md"] if (out_dir / n).exists()]


def _unsupported(spec, code, msg, meta, cav):
    return DeepLensWaveOpticsProbeResult(run_id=spec.run_id, status="unsupported", candidate=spec.candidate, error_code=code, error_message=msg, caveats=[*cav, msg], metadata=meta)


def _failed(spec, code, msg, meta, cav, loss_before=None, gradient_norm=None):
    return DeepLensWaveOpticsProbeResult(run_id=spec.run_id, status="failed", candidate=spec.candidate, error_code=code, error_message=msg, loss_before=loss_before, optical_gradient_norm=gradient_norm, caveats=[*cav, msg], metadata=meta)
