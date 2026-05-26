"""DeepLens Trainable Parameter Inspection for Phase 54."""

from __future__ import annotations
from typing import Any


def inspect_deeplens_trainable_parameters(backend_id: str = "deeplens_geolens_geometric",
                                           lens_file: str = "auto:cooke",
                                           device: str = "cpu") -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "needs_followup", "evidence_level": "diagnostic_evidence",
        "parameter_count": 0, "trainable_count": 0,
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

        import torch, importlib
        geolens_mod = importlib.import_module("deeplens.geolens")
        geolens = geolens_mod.GeoLens(resolution.resolved_path, device=device)
        params = list(geolens.parameters())
        result["parameter_count"] = len(params)
        trainable = [p for p in params if p.requires_grad]
        result["trainable_count"] = len(trainable)
        points = torch.tensor([[0.0, 0.0]], device=device)
        wvln = torch.tensor([0.55], device=device)
        ks = torch.tensor([[0.0, 0.0]], device=device)
        psf = geolens.psf(points, wvln, ks, model="geometric")
        if psf.requires_grad:
            psf.sum().backward()
            for i, p in enumerate(trainable):
                gn = float(p.grad.norm().item()) if p.grad is not None else 0.0
                if gn > 1000:
                    result["unstable_gradient_parameters"].append(i)
                elif gn > 0:
                    result["top_gradient_parameters"].append(i)
                else:
                    result["zero_gradient_parameters"].append(i)
        if result["trainable_count"] == 0:
            result["diagnosis"] = "no_trainable_parameters"
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
