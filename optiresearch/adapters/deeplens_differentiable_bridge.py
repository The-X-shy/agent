"""Differentiable bridges from DeepLens surface components to torch PSF tensors.

Phase 20: Connects Fresnel DOE and Binary2Phase surfaces to the HSI proxy
loss pipeline via a differentiable phase-to-PSF transform.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any

import torch

SURFACE_MODULES: dict[str, str] = {
    "Fresnel": "deeplens.diffractive_surface.fresnel",
    "Binary2Phase": "deeplens.phase_surface.binary2",
}

SURFACE_TRAINABLE_NAMES: dict[str, list[str]] = {
    "Fresnel": ["f0"],
    "Binary2Phase": ["d", "order2", "order4", "order6", "order8", "order10", "order12"],
}


class FresnelHSIBridge:
    """Differentiable bridge: Fresnel DOE -> PSF tensor via FFT proxy.

    Path: Fresnel.f0 -> phase_func() -> phase map -> field = exp(1j*phase)
          -> FFT2 -> |.|^2 -> normalize -> PSF

    realization_level: native_component_proxy
    component_native_grad: true (Fresnel.f0 is a native DeepLens parameter)
    full_lens_native_psf: false (using phase-to-PSF FFT proxy, not full wave prop)
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.surface: Any = None
        self.optimizer: Any = None
        self.realization_level = "native_component_proxy"
        self.component_native_grad = True
        self.full_lens_native_psf = False

    def build_component(self) -> Any:
        """Import and instantiate a Fresnel DOE surface."""
        repo_path = os.getenv("DEEPLENS_REPO_PATH")
        if repo_path and Path(repo_path).is_dir() and repo_path not in sys.path:
            sys.path.insert(0, repo_path)

        module_path = SURFACE_MODULES["Fresnel"]
        mod = importlib.import_module(module_path)
        cls = getattr(mod, "Fresnel")
        surface = cls(d=0.0, f0=50.0, res=48, device=self.device)
        self.surface = surface
        return surface

    def psf_from_component_torch(
        self, surface: Any | None = None, num_bands: int = 4, psf_size: int = 16
    ) -> torch.Tensor:
        """Generate a [B, K, K] differentiable PSF cube from Fresnel phase."""
        s = surface or self.surface
        if s is None:
            raise RuntimeError("No surface built. Call build_component() first.")

        phase = self._get_phase(s)

        psfs = []
        for b in range(num_bands):
            wavelength_scale = 1.0 + 0.02 * (b - num_bands / 2)
            scaled_phase = phase * wavelength_scale
            field = torch.exp(1j * scaled_phase)
            psf_full = torch.abs(torch.fft.fft2(field)) ** 2
            psf_full = torch.fft.fftshift(psf_full)
            h, w = psf_full.shape
            ch, cw = h // 2, w // 2
            kh, kw = psf_size // 2, psf_size // 2
            psf_cropped = psf_full[ch - kh : ch - kh + psf_size, cw - kw : cw - kw + psf_size]
            psf_normalized = psf_cropped / (psf_cropped.sum() + 1e-8)
            psfs.append(psf_normalized)

        return torch.stack(psfs, dim=0)

    def _get_phase(self, surface: Any) -> torch.Tensor:
        if callable(getattr(surface, "phase_func", None)):
            return surface.phase_func()
        raise RuntimeError("Fresnel surface has no phase_func()")

    def get_trainable_parameters(self) -> list[torch.nn.Parameter]:
        params = []
        for name in SURFACE_TRAINABLE_NAMES["Fresnel"]:
            val = getattr(self.surface, name, None)
            if isinstance(val, torch.nn.Parameter):
                params.append(val)
            elif isinstance(val, torch.Tensor) and val.requires_grad:
                params.append(val)
        return params

    def get_optimizer(self, learning_rate: float = 1e-3) -> Any:
        s = self.surface
        if callable(getattr(s, "get_optimizer", None)):
            for args, kwargs in [
                ((), {"lr": learning_rate}),
                ((learning_rate,), {}),
                ((), {}),
            ]:
                try:
                    opt = s.get_optimizer(*args, **kwargs)
                    if opt is not None:
                        self.optimizer = opt
                        return opt
                except Exception:
                    continue

        if callable(getattr(s, "get_optimizer_params", None)):
            for args, kwargs in [
                ((), {"lr": learning_rate}),
                ((learning_rate,), {}),
                ((), {}),
            ]:
                try:
                    params = s.get_optimizer_params(*args, **kwargs)
                    self.optimizer = torch.optim.Adam(params)
                    return self.optimizer
                except Exception:
                    continue

        raise RuntimeError("Fresnel surface has no get_optimizer or get_optimizer_params")

    def parameter_snapshot(self) -> dict[str, Any]:
        snapshot: dict[str, Any] = {}
        for name in SURFACE_TRAINABLE_NAMES["Fresnel"]:
            val = getattr(self.surface, name, None)
            if val is not None:
                tensor = val.detach().cpu()
                if tensor.numel() == 1:
                    snapshot[name] = float(tensor.item())
                else:
                    snapshot[name] = {"shape": list(tensor.shape), "norm": float(tensor.norm().item())}
        return snapshot

    def gradient_norm(self) -> float:
        total = 0.0
        for param in self.get_trainable_parameters():
            if param.grad is not None:
                total += float(param.grad.detach().norm().cpu().item() ** 2)
        return float(total ** 0.5)


class Binary2PhaseHSIBridge:
    """Differentiable bridge: Binary2Phase -> PSF tensor via FFT proxy.

    Path: Binary2Phase.{d, order2, ..., order12} -> phi(x,y) -> phase map
          -> field = exp(1j*phase) -> FFT2 -> |.|^2 -> normalize -> PSF
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.surface: Any = None
        self.optimizer: Any = None
        self.realization_level = "native_component_proxy"
        self.component_native_grad = True
        self.full_lens_native_psf = False

    def build_component(self) -> Any:
        repo_path = os.getenv("DEEPLENS_REPO_PATH")
        if repo_path and Path(repo_path).is_dir() and repo_path not in sys.path:
            sys.path.insert(0, repo_path)

        module_path = SURFACE_MODULES["Binary2Phase"]
        mod = importlib.import_module(module_path)
        cls = getattr(mod, "Binary2Phase")
        surface = cls(
            r=5.0, d=0.0,
            order2=1.0, order4=0.2, order6=0.05,
            order8=0.0, order10=0.0, order12=0.0,
            device=self.device,
        )
        self.surface = surface
        return surface

    def psf_from_component_torch(
        self, surface: Any | None = None, num_bands: int = 4, psf_size: int = 16
    ) -> torch.Tensor:
        s = surface or self.surface
        if s is None:
            raise RuntimeError("No surface built. Call build_component() first.")

        phase = self._get_phase(s)
        psfs = []
        for b in range(num_bands):
            wavelength_scale = 1.0 + 0.02 * (b - num_bands / 2)
            scaled_phase = phase * wavelength_scale
            field = torch.exp(1j * scaled_phase)
            psf_full = torch.abs(torch.fft.fft2(field)) ** 2
            psf_full = torch.fft.fftshift(psf_full)
            h, w = psf_full.shape
            ch, cw = h // 2, w // 2
            kh, kw = psf_size // 2, psf_size // 2
            psf_cropped = psf_full[ch - kh : ch - kh + psf_size, cw - kw : cw - kw + psf_size]
            psf_normalized = psf_cropped / (psf_cropped.sum() + 1e-8)
            psfs.append(psf_normalized)
        return torch.stack(psfs, dim=0)

    def _get_phase(self, surface: Any) -> torch.Tensor:
        if callable(getattr(surface, "phi", None)):
            coords = torch.linspace(-1.0, 1.0, 48, device=self.device)
            y, x = torch.meshgrid(coords, coords, indexing="ij")
            return surface.phi(x, y)
        raise RuntimeError("Binary2Phase surface has no phi(x, y)")

    def get_trainable_parameters(self) -> list[torch.nn.Parameter]:
        params = []
        for name in SURFACE_TRAINABLE_NAMES["Binary2Phase"]:
            val = getattr(self.surface, name, None)
            if isinstance(val, torch.nn.Parameter):
                params.append(val)
            elif isinstance(val, torch.Tensor) and val.requires_grad:
                params.append(val)
        return params

    def get_optimizer(self, learning_rate: float = 1e-3) -> Any:
        s = self.surface
        if callable(getattr(s, "get_optimizer", None)):
            for args, kwargs in [
                ((), {"lr": learning_rate}),
                ((), {"lrs": [learning_rate, learning_rate]}),
                ((), {}),
            ]:
                try:
                    opt = s.get_optimizer(*args, **kwargs)
                    if opt is not None:
                        self.optimizer = opt
                        return opt
                except Exception:
                    continue

        if callable(getattr(s, "get_optimizer_params", None)):
            for args, kwargs in [
                ((), {"lr": learning_rate}),
                ((), {"lrs": [learning_rate, learning_rate]}),
                ((), {}),
            ]:
                try:
                    params = s.get_optimizer_params(*args, **kwargs)
                    self.optimizer = torch.optim.Adam(params)
                    return self.optimizer
                except Exception:
                    continue

        raise RuntimeError("Binary2Phase has no get_optimizer or get_optimizer_params")

    def parameter_snapshot(self) -> dict[str, Any]:
        snapshot: dict[str, Any] = {}
        for name in SURFACE_TRAINABLE_NAMES["Binary2Phase"]:
            val = getattr(self.surface, name, None)
            if val is not None:
                tensor = val.detach().cpu()
                if tensor.numel() == 1:
                    snapshot[name] = float(tensor.item())
                else:
                    snapshot[name] = {"shape": list(tensor.shape), "norm": float(tensor.norm().item())}
        return snapshot

    def gradient_norm(self) -> float:
        total = 0.0
        for param in self.get_trainable_parameters():
            if param.grad is not None:
                total += float(param.grad.detach().norm().cpu().item() ** 2)
        return float(total ** 0.5)
