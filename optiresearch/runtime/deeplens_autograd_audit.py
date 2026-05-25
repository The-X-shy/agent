"""DeepLens Autograd Audit for Phase 54."""

from __future__ import annotations
from typing import Any


def run_deeplens_autograd_audit(backend_id: str = "deeplens_geolens_geometric",
                                 lens_file: str = "auto:cooke", device: str = "cpu") -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "needs_followup",
        "evidence_level": "diagnostic_evidence",
        "trainable_param_count": 0, "params_with_grad": 0,
        "grad_norm_max": 0.0, "graph_connected": False,
        "detach_suspected": False, "psf_requires_grad": False,
        "loss_requires_grad": False, "diagnosis": [], "recommended_next_strategy": "",
    }
    try:
        import torch, importlib
        geolens_mod = importlib.import_module("deeplens.geolens")
        lens_path = _find_lens(lens_file)
        if lens_path is None:
            result["diagnosis"] = ["lens_file_not_found"]
            result["status"] = "unavailable"
            return result
        geolens = geolens_mod.GeoLens(lens_path, device=device)
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
            result["graph_connected"] = result["grad_norm_max"] > 0.0
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


def _find_lens(lens_name: str) -> str | None:
    from pathlib import Path
    for p in [Path(f"/Users/lilin/Desktop/external/DeepLens/datasets/lenses/{lens_name}.json"),
              Path(f"/mnt/d/external/DeepLens/datasets/lenses/{lens_name}.json")]:
        if p.exists():
            return str(p)
    return None
