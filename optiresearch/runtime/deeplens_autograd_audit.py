"""DeepLens Autograd Audit for Phase 54."""

from __future__ import annotations
from typing import Any

import importlib

import torch

from optiresearch.adapters.deeplens_geolens_params import (
    DEFAULT_GEOLENS_LRS,
    activate_geolens_trainable_parameters,
)


def run_deeplens_autograd_audit(backend_id: str = "deeplens_geolens_geometric",
                                 lens_file: str = "auto:cooke", device: str = "cpu") -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "needs_followup",
        "evidence_level": "diagnostic_evidence",
        "parameter_count": 0,
        "trainable_param_count": 0, "params_with_grad": 0,
        "nonzero_grad_param_count": 0,
        "grad_norm_max": 0.0, "grad_norm_mean": 0.0,
        "graph_connected": False,
        "detach_suspected": False, "psf_requires_grad": False,
        "loss_requires_grad": False, "candidate_update_changes_parameter": False,
        "diagnosis": [], "recommended_next_strategy": "",
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
            result["diagnosis"] = ["lens_file_not_found"]
            result["status"] = "unavailable"
            result["error_code"] = "LENS_FILE_NOT_FOUND"
            return result
        result["resolved_lens_file"] = resolution.resolved_path
        result["lens_resolution_source"] = resolution.source

        geolens_mod = importlib.import_module("deeplens.geolens")
        geolens = geolens_mod.GeoLens(resolution.resolved_path, device=device)

        param_groups, trainable = activate_geolens_trainable_parameters(geolens, lrs=DEFAULT_GEOLENS_LRS)
        result["parameter_count"] = len(trainable)
        result["trainable_param_count"] = len(trainable)
        if not trainable:
            result["diagnosis"] = ["no_native_trainable_parameters"]
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

        result["psf_requires_grad"] = bool(psf.requires_grad)
        loss = (psf * psf).sum()
        result["loss_requires_grad"] = bool(loss.requires_grad)
        if loss.requires_grad:
            param_before = [p.detach().clone() for p in trainable]
            loss.backward()
            grad_norms = [float(p.grad.norm().item()) for p in trainable if p.grad is not None]
            result["params_with_grad"] = len(grad_norms)
            result["nonzero_grad_param_count"] = len([gn for gn in grad_norms if gn > 0.0])
            result["grad_norm_max"] = max(grad_norms) if grad_norms else 0.0
            result["grad_norm_mean"] = sum(grad_norms) / len(grad_norms) if grad_norms else 0.0
            result["graph_connected"] = result["grad_norm_max"] > 0.0
            if result["graph_connected"]:
                try:
                    if callable(getattr(geolens, "get_optimizer", None)):
                        optimizer = geolens.get_optimizer(lrs=DEFAULT_GEOLENS_LRS, optim_mat=False)
                    else:
                        optimizer = torch.optim.SGD(param_groups or trainable, lr=1e-6)
                    optimizer.step()
                    result["candidate_update_changes_parameter"] = any(
                        not torch.allclose(before, after.detach(), rtol=0.0, atol=0.0)
                        for before, after in zip(param_before, trainable)
                    )
                except Exception as exc:
                    result["diagnosis"].append(f"candidate_update_failed: {exc}")
        if not result["psf_requires_grad"]:
            result["diagnosis"].append("psf_not_differentiable")
            result["detach_suspected"] = True
        if not result["graph_connected"]:
            result["diagnosis"].append("gradient_flow_blocked")
            result["recommended_next_strategy"] = "component_first_probe"
        if result["graph_connected"]:
            result["status"] = "succeeded"
            result["recommended_next_strategy"] = "geolens_curriculum_probe"
    except Exception as e:
        result["diagnosis"] = [f"audit_exception: {e}"]
        result["status"] = "unavailable"
    return result
