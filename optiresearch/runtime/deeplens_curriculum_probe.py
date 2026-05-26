"""DeepLens Curriculum Probe for Phase 54."""

from __future__ import annotations
from typing import Any


def run_deeplens_curriculum_probe(max_steps: int = 3, device: str = "cpu",
                                   lens_file: str | None = None,
                                   backend_id: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "succeeded", "evidence_level": "diagnostic_evidence",
        "curriculum_progress": 0, "stages_completed": 0, "stage_results": [],
        "failure_stage": None, "claim_ceiling": "diagnostic_evidence",
        "requested_lens_file": lens_file,
        "resolved_lens_file": None,
        "lens_resolution_source": None,
        "checked_lens_paths": [],
    }
    if lens_file is not None:
        try:
            from optiresearch.optics.lens_file_resolver import resolve_lens_file
            resolution = resolve_lens_file(lens_file=lens_file, backend_id=backend_id)
            result["checked_lens_paths"] = resolution.checked_paths
            if resolution.exists:
                result["resolved_lens_file"] = resolution.resolved_path
                result["lens_resolution_source"] = resolution.source
        except Exception:
            pass
    try:
        import torch
        from optiresearch.runtime.lightweight_experiments import (
            _generate_synthetic_hsi_torch, _generate_proxy_psf_torch, _LinearReconstructor,
        )
        hsi = _generate_synthetic_hsi_torch(4, 16, device)
        _, phase_mask = _generate_proxy_psf_torch(4, 15, device)
        recon = _LinearReconstructor(4, 16, device)
        stages = [
            {"name": "psf_diagnostic", "train_optical": False, "train_recon": False},
            {"name": "recon_only", "train_optical": False, "train_recon": True},
            {"name": "unfreeze_one_param", "train_optical": True, "train_recon": True},
            {"name": "joint_tiny_update", "train_optical": True, "train_recon": True},
        ]
        for stage_idx, stage in enumerate(stages):
            stage_result = {"stage": stage["name"], "loss": None, "grad_norm": 0.0}
            try:
                loss_val = _run_stage(hsi, phase_mask, recon, stage, max_steps, device)
                stage_result["loss"] = float(loss_val) if loss_val else None
            except Exception as e:
                stage_result["error"] = str(e)
                result["failure_stage"] = stage["name"]
                break
            result["stage_results"].append(stage_result)
            result["stages_completed"] = stage_idx + 1
        result["curriculum_progress"] = result["stages_completed"]
    except Exception:
        result["status"] = "unavailable"
    return result


def _run_stage(hsi, phase_mask, recon, stage, steps, device):
    import torch
    opt_params = []
    if stage["train_optical"]:
        opt_params.append(phase_mask)
    if stage["train_recon"]:
        opt_params.extend(recon.parameters())
    if not opt_params:
        return None
    opt = torch.optim.Adam(opt_params, lr=1e-6)
    for _ in range(steps):
        band_offsets = torch.linspace(0, 6.28, 4, device=device)
        psfs = [torch.abs(torch.fft.fft2(torch.exp(1j * (phase_mask + bo)))) ** 2 for bo in band_offsets]
        psf_cube = torch.stack([p / (p.sum() + 1e-8) for p in psfs])
        measured = torch.stack([
            torch.fft.ifft2(torch.fft.fft2(hsi[b]) * torch.fft.fft2(psf_cube[b], s=hsi[b].shape[-2:])).real
            for b in range(4)
        ])
        loss = torch.nn.MSELoss()(recon(measured), hsi)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss.item()
