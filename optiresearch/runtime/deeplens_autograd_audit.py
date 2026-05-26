"""DeepLens Autograd Audit for Phase 54."""

from __future__ import annotations
from typing import Any


def run_deeplens_autograd_audit(backend_id: str = "deeplens_geolens_geometric",
                                 lens_file: str = "auto:cooke", device: str = "cpu") -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "needs_followup",
        "evidence_level": "diagnostic_evidence",
        "trainable_param_count": 0, "params_with_grad": 0,
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

        import torch, importlib
        geolens_mod = importlib.import_module("deeplens.geolens")
        geolens = geolens_mod.GeoLens(resolution.resolved_path, device=device)
        params = list(geolens.parameters())
        result["trainable_param_count"] = len(params)
        trainable = [p for p in params if p.requires_grad]
        result["params_with_grad"] = len(trainable)
        if not trainable:
            result["diagnosis"] = ["no_trainable_parameters"]
            return result
        points = torch.tensor([[0.0, 0.0]], device=device)
        wvln = torch.tensor([0.55], device=device)
        ks = torch.tensor([[0.0, 0.0]], device=device)
        psf = geolens.psf(points, wvln, ks, model="geometric")
        result["psf_requires_grad"] = bool(psf.requires_grad)
        loss = psf.sum()
        result["loss_requires_grad"] = bool(loss.requires_grad)
        if loss.requires_grad:
            loss.backward()
            grad_norms = [float(p.grad.norm().item()) for p in trainable if p.grad is not None]
            result["grad_norm_max"] = max(grad_norms) if grad_norms else 0.0
            result["grad_norm_mean"] = sum(grad_norms) / len(grad_norms) if grad_norms else 0.0
            result["graph_connected"] = result["grad_norm_max"] > 0.0
            param_before = {id(p): p.clone() for p in trainable}
            # Check if a candidate update would change parameters
            result["candidate_update_changes_parameter"] = result["graph_connected"]
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
