"""DeepLens Trainable Parameter Inspection for Phase 54."""

from __future__ import annotations
from typing import Any

import importlib

import torch

from optiresearch.adapters.deeplens_geolens_params import (
    DEFAULT_GEOLENS_LRS,
    activate_geolens_trainable_parameters,
)


def inspect_deeplens_trainable_parameters(backend_id: str = "deeplens_geolens_geometric",
                                           lens_file: str = "auto:cooke",
                                           device: str = "cpu") -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "needs_followup", "evidence_level": "diagnostic_evidence",
        "parameter_count": 0, "trainable_count": 0,
        "params_with_grad": 0, "grad_norm_max": 0.0, "grad_norm_mean": 0.0,
        "psf_requires_grad": False, "loss_requires_grad": False,
        "graph_connected": False,
        "surface_groups": {}, "top_gradient_parameters": [],
        "zero_gradient_parameters": [], "unstable_gradient_parameters": [],
        "recommended_trainable_subset": [], "recommended_strategy": "",
        "requested_lens_file": lens_file,
        "resolved_lens_file": None,
        "lens_resolution_source": None,
        "checked_lens_paths": [],
    }
    try:
        from optiresearch.optics.lens_file_resolver import resolve_lens_file
        resolution = resolve_lens_file(lens_file=lens_file, backend_id=backend_id)
        result["checked_lens_paths"] = resolution.checked_paths
        if not resolution.exists:
            result["status"] = "unavailable"
            result["error_code"] = "LENS_FILE_NOT_FOUND"
            result["diagnosis"] = "lens_file_not_found"
            return result
        result["resolved_lens_file"] = resolution.resolved_path
        result["lens_resolution_source"] = resolution.source

        geolens_mod = importlib.import_module("deeplens.geolens")
        geolens = geolens_mod.GeoLens(resolution.resolved_path, device=device)

        _, trainable = activate_geolens_trainable_parameters(geolens, lrs=DEFAULT_GEOLENS_LRS)
        result["parameter_count"] = len(trainable)
        result["trainable_count"] = len(trainable)
        if not trainable:
            result["diagnosis"] = "no_native_trainable_parameters"
            return result

        for p in trainable:
            p.grad = None

        points = torch.tensor([[0.0, 0.0, -10000.0]], device=device, dtype=torch.float32)
        orig_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float32)
            psf = geolens.psf(points, wvln=0.55, ks=9, model="geometric")
        finally:
            torch.set_default_dtype(orig_dtype)

        result["psf_requires_grad"] = bool(getattr(psf, "requires_grad", False))
        loss = (psf * psf).sum()
        result["loss_requires_grad"] = bool(getattr(loss, "requires_grad", False))
        if result["loss_requires_grad"]:
            loss.backward()
            grad_norms: list[float] = []
            for i, p in enumerate(trainable):
                gn = float(p.grad.norm().item()) if p.grad is not None else 0.0
                if gn > 0:
                    grad_norms.append(gn)
                if gn > 1000:
                    result["unstable_gradient_parameters"].append(i)
                elif gn > 0:
                    result["top_gradient_parameters"].append(i)
                else:
                    result["zero_gradient_parameters"].append(i)
            result["params_with_grad"] = len(grad_norms)
            result["grad_norm_max"] = max(grad_norms) if grad_norms else 0.0
            result["grad_norm_mean"] = sum(grad_norms) / len(grad_norms) if grad_norms else 0.0
            result["graph_connected"] = result["grad_norm_max"] > 0.0

        if result["trainable_count"] == 0:
            result["diagnosis"] = "no_native_trainable_parameters"
        elif result["zero_gradient_parameters"] and not result["top_gradient_parameters"]:
            result["recommended_strategy"] = "autograd_audit"
        elif result["unstable_gradient_parameters"]:
            result["recommended_strategy"] = "surface_freeze_unfreeze"
        else:
            result["status"] = "succeeded"
            result["recommended_strategy"] = "geolens_curriculum_probe"
    except Exception as e:
        result["status"] = "unavailable"
        result["error_code"] = str(e)[:200]
    return result
