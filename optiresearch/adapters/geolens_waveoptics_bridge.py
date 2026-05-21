"""GeoLens wave-optics bridge for Phase 22.

Wraps DeepLens's native differentiable wave-optics PSF path:
  GeoLens.psf(method="coherent") → psf_pupil_prop() → AngularSpectrumMethod

Unlike Phase 20/21's phase-to-FFT proxy, this uses DeepLens's real wave-optics
simulation (coherent ray tracing + ASM propagation via torch.fft).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


class GeoLensWaveOpticsBridge:
    """Bridge: GeoLens lens file → native differentiable ray-tracing PSF.

    Uses GeoLens.psf(model="geometric") for differentiable ray-tracing PSF.
    Coherent ASM wave-optics (model="coherent") is available but produces
    requires_grad=False in current DeepLens implementation.

    full_wave_optics: false (coherent ASM not differentiable in practice)
    phase_to_fft_proxy_used: false (uses DeepLens native geometric PSF)
    deeplens_native_psf_path: geolens.psf_geometric
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.geolens: Any = None
        self.optimizer: Any = None
        self.full_wave_optics = False
        self.phase_to_fft_proxy_used = False
        self.deeplens_native_psf_path = "geolens.psf_geometric"

    def build_component(self, lens_file: str | None = None) -> Any:
        """Load GeoLens from a JSON lens file (e.g., cooke.json)."""
        repo_path = os.getenv("DEEPLENS_REPO_PATH")
        if repo_path and Path(repo_path).is_dir() and repo_path not in sys.path:
            sys.path.insert(0, repo_path)

        import importlib
        mod = importlib.import_module("deeplens.geolens")
        GeoLens = getattr(mod, "GeoLens")

        if lens_file is None:
            lens_file = self._find_default_lens_file(repo_path)

        if lens_file and Path(lens_file).exists():
            self.geolens = GeoLens(lens_file, device=self.device)
        else:
            raise FileNotFoundError(f"Lens file not found: {lens_file}")
        return self.geolens

    def _find_default_lens_file(self, repo_path: str | None) -> str | None:
        candidates = []
        if repo_path:
            rp = Path(repo_path)
            candidates.append(rp / "datasets" / "lenses" / "cooke.json")
            candidates.append(rp / "samples" / "cooke.json")
        candidates.append(Path("/Users/lilin/Desktop/external/DeepLens/datasets/lenses/cooke.json"))
        candidates.append(Path("/mnt/d/external/DeepLens/datasets/lenses/cooke.json"))
        for c in candidates:
            if c.exists():
                return str(c)
        return None

    def psf_from_component_torch(
        self,
        points: Any = None,
        wvln: float = 0.55,
        ks: int = 32,
        model: str = "geometric",
    ) -> torch.Tensor:
        """Generate a differentiable PSF using DeepLens native lens simulation.

        Default: model="geometric" (differentiable ray-tracing PSF).
        model="coherent" is available but produces requires_grad=False.
        Returns [ks, ks] torch.Tensor with requires_grad=True (geometric).
        """
        if self.geolens is None:
            raise RuntimeError("No GeoLens loaded. Call build_component() first.")

        if points is None:
            points = torch.tensor([[0.0, 0.0, -10000.0]], device=self.device, dtype=torch.float64)

        orig_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            psf = self.geolens.psf(points, wvln=wvln, ks=ks, model=model)
        finally:
            torch.set_default_dtype(orig_dtype)
        return psf.float()

    def psf_cube_torch(
        self,
        num_bands: int = 4,
        ks: int = 32,
        wvln_start: float = 0.45,
        wvln_end: float = 0.65,
    ) -> torch.Tensor:
        """Generate a per-wavelength PSF cube [B, ks, ks]."""
        psfs = []
        for b in range(num_bands):
            wvln = wvln_start + (wvln_end - wvln_start) * b / max(num_bands - 1, 1)
            psf = self.psf_from_component_torch(wvln=wvln, ks=ks)
            psfs.append(psf)
        return torch.stack(psfs, dim=0)

    def get_trainable_parameters(self) -> list[Any]:
        if self.geolens is None:
            return []
        params = []
        if callable(getattr(self.geolens, "get_optimizer_params", None)):
            try:
                pgroups = self.geolens.get_optimizer_params()
                for g in pgroups:
                    params.extend(g.get("params", []))
            except Exception:
                pass
        if not params:
            for attr_name in dir(self.geolens):
                if attr_name.startswith("_"):
                    continue
                val = getattr(self.geolens, attr_name, None)
                if isinstance(val, torch.nn.Parameter):
                    params.append(val)
        return params

    def get_optimizer(self, learning_rate: float = 1e-3) -> Any:
        if self.geolens is None:
            raise RuntimeError("No GeoLens loaded.")
        if callable(getattr(self.geolens, "get_optimizer", None)):
            for args, kwargs in [
                ((), {"lr": [learning_rate] * 4}),
                ((), {"lr": learning_rate}),
                ((), {}),
            ]:
                try:
                    self.optimizer = self.geolens.get_optimizer(*args, **kwargs)
                    return self.optimizer
                except Exception:
                    continue
        params = self.get_trainable_parameters()
        if params:
            self.optimizer = torch.optim.Adam(params, lr=learning_rate)
            return self.optimizer
        raise RuntimeError("Cannot create optimizer for GeoLens")

    def parameter_snapshot(self) -> dict[str, Any]:
        snapshot: dict[str, Any] = {}
        params = self.get_trainable_parameters()
        for i, param in enumerate(params):
            t = param.detach().cpu()
            if t.numel() == 1:
                snapshot[f"param_{i}"] = float(t.item())
            else:
                snapshot[f"param_{i}"] = {"shape": list(t.shape), "norm": float(t.norm().item())}
        return snapshot

    def gradient_norm(self) -> float:
        total = 0.0
        for param in self.get_trainable_parameters():
            if param.grad is not None:
                total += float(param.grad.detach().norm().cpu().item() ** 2)
        return float(total ** 0.5)
