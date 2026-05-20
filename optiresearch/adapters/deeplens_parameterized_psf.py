"""DeepLens-backed parameterized PSF generator.

Maps optical co-design variables to DeepLens lens/PSF configuration.
When DeepLens is unavailable or variables are unsupported,
returns structured fallback with explicit caveats.

IMPORTANT: This is BLACK-BOX optimization, NOT differentiable.
Native parameter update is false. Results must not be claimed
as native DeepLens optimization.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np


class DeepLensParameterizedPSFGenerator:
    """Generate PSF cubes from optical variables using DeepLens backend.

    Maps 5 optical variables to DeepLens config:
    - surface_curvature → f_number, focal_length (via ParaxialLens)
    - chromatic_shift → wavelength-dependent depth sampling
    - depth_variation → depth range/spacing scaling
    - phase_mask_strength → UNSUPPORTED (no DOE/phase API)
    - doe_grating_period → UNSUPPORTED (no diffraction API)
    """

    def __init__(
        self,
        experiment_spec: dict[str, Any] | None = None,
        capabilities: dict[str, Any] | None = None,
        deeplens_module: Any = None,
        strict_deeplens: bool = False,
    ) -> None:
        self._spec = experiment_spec or {}
        self._strict = strict_deeplens
        self._repo_path: str | None = None
        self._import_path: str | None = None
        self._is_source_checkout: bool = False

        # Support DEEPLENS_REPO_PATH
        repo_path = os.getenv("DEEPLENS_REPO_PATH", "")
        if repo_path and Path(repo_path).is_dir() and (Path(repo_path) / "deeplens").is_dir():
            self._repo_path = repo_path
            self._is_source_checkout = True
            if repo_path not in sys.path:
                sys.path.insert(0, repo_path)

        self._capabilities = capabilities or self._probe_capabilities()
        self._deeplens = deeplens_module
        if self._deeplens is None:
            try:
                import deeplens as dl
                self._deeplens = dl
                if hasattr(dl, "__file__") and dl.__file__:
                    self._import_path = dl.__file__
                if self._repo_path and self._import_path and self._repo_path in self._import_path:
                    self._is_source_checkout = True
            except ImportError:
                self._deeplens = None

    @property
    def deeplens_available(self) -> bool:
        return self._deeplens is not None

    def supported_variables(self) -> dict[str, dict[str, Any]]:
        """Return which variables are supported and how they map."""
        return {
            "surface_curvature": {
                "supported": True,
                "maps_to": "f_number",
                "description": "Controls ParaxialLens f_number (lower = more curvature = tighter focus)",
                "range": [0.0, 1.0],
                "mapping_formula": "f_number = 8.0 - curvature * 6.4  → range [1.6, 8.0]",
            },
            "chromatic_shift": {
                "supported": True,
                "maps_to": "wavelength_depth_offset",
                "description": "Wavelength-dependent depth sampling offset",
                "range": [0.0, 1.0],
                "mapping_formula": "per-wavelength depth shift proportional to chromatic_shift",
            },
            "depth_variation": {
                "supported": True,
                "maps_to": "depth_range_scaling",
                "description": "Scales the depth sampling range",
                "range": [0.0, 1.0],
                "mapping_formula": "depth_range = base_range * (0.5 + depth_variation)",
            },
            "phase_mask_strength": {
                "supported": False,
                "maps_to": None,
                "description": "Wavefront phase modulation — requires DOE/phase API not available in ParaxialLens",
                "fallback": "adapter_proxy",
            },
            "doe_grating_period": {
                "supported": False,
                "maps_to": None,
                "description": "Diffractive optical element — requires diffraction API not available in ParaxialLens",
                "fallback": "adapter_proxy",
            },
        }

    def unsupported_variables(self, optical_vars: dict[str, float]) -> list[dict[str, Any]]:
        """List which variables in the current set are unsupported."""
        supported = self.supported_variables()
        unsupported = []
        for name, value in optical_vars.items():
            info = supported.get(name, {})
            if not info.get("supported", False):
                unsupported.append({
                    "variable": name,
                    "value": value,
                    "reason": info.get("description", "Unknown variable"),
                    "fallback": info.get("fallback", "ignored"),
                })
        return unsupported

    def map_variables_to_deeplens_config(
        self,
        optical_vars: dict[str, float],
    ) -> dict[str, Any]:
        """Map optical variables to a DeepLens-compatible config.

        Returns a dict with:
        - deeplens_config: the translated config for DeepLens
        - unsupported_variables: list of variables that could not be mapped
        - parameter_mapping: how each variable was mapped
        """
        curvature = float(optical_vars.get("surface_curvature", 0.5))
        chroma = float(optical_vars.get("chromatic_shift", 0.3))
        depth_var = float(optical_vars.get("depth_variation", 0.5))
        phase_mask = float(optical_vars.get("phase_mask_strength", 0.5))
        doe = float(optical_vars.get("doe_grating_period", 1.0))

        # Map surface_curvature to f_number: higher curvature → lower f_number
        f_number = round(8.0 - curvature * 6.4, 2)  # range [1.6, 8.0]
        focal_length = round(30.0 + curvature * 40.0, 1)  # range [30, 70] mm

        # Map depth_variation to depth range (DeepLens uses negative convention)
        base_depths = [0.0, -5.0, -10.0, -15.0, -20.0]
        depth_scale = 0.5 + depth_var  # range [0.5, 1.5]
        depths_mm = [d * depth_scale for d in base_depths]

        # Map chromatic_shift to wavelength-dependent offsets
        wavelength_count = 31
        wavelengths_nm = list(np.linspace(450.0, 750.0, wavelength_count))

        unsupported = self.unsupported_variables(optical_vars)

        parameter_mapping = {
            "surface_curvature": {
                "input": curvature,
                "maps_to": "f_number",
                "output": f_number,
                "also_maps_to": {"focal_length": focal_length},
            },
            "depth_variation": {
                "input": depth_var,
                "maps_to": "depth_range_scaling",
                "output": depth_scale,
                "depths_mm": depths_mm,
            },
            "chromatic_shift": {
                "input": chroma,
                "maps_to": "wavelength_depth_offset",
                "output": chroma,
                "wavelength_count": wavelength_count,
            },
        }
        for u in unsupported:
            parameter_mapping[u["variable"]] = {
                "input": optical_vars.get(u["variable"]),
                "maps_to": None,
                "output": None,
                "unsupported": True,
                "reason": u["reason"],
            }

        config = {
            "config_type": "DeepLensCandidateConfig",
            "backend": "deeplens",
            "optical_parameters": {
                "focal_length": focal_length,
                "f_number": f_number,
                "aperture": round(focal_length / f_number, 2),
            },
            "wavelengths_nm": wavelengths_nm,
            "depths_mm": depths_mm,
            "psf_size": 32,
            "sensor_type": "mock",
            "chromatic_shift": chroma,
            "wavelength_count": wavelength_count,
            "encoder_type": "conventional",  # base PSF, encoder transform applied separately
        }

        return {
            "deeplens_config": config,
            "unsupported_variables": unsupported,
            "parameter_mapping": parameter_mapping,
            "differentiable": False,
            "native_parameter_update": False,
            "optimization_mode": "black_box",
        }

    def generate_psf_cube(
        self,
        optical_vars: dict[str, float],
        output_dir: Path,
    ) -> dict[str, Any]:
        """Generate a PSF cube from optical variables via DeepLens.

        Returns a dict with:
        - status: "succeeded", "partial", "unsupported", "fallback"
        - psf_cube: np.ndarray or None
        - psf_path: Path or None
        - unsupported_variables: list
        - fallback_used: bool
        - metadata: dict
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        mapping = self.map_variables_to_deeplens_config(optical_vars)

        # Save mapping
        (output_dir / "deeplens_parameter_mapping.json").write_text(
            json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        unsupported = mapping["unsupported_variables"]
        (output_dir / "unsupported_variables.json").write_text(
            json.dumps(unsupported, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        has_unsupported = len(unsupported) > 0

        if not self.deeplens_available:
            if self._strict:
                return {
                    "status": "unavailable",
                    "psf_cube": None,
                    "psf_path": None,
                    "unsupported_variables": unsupported,
                    "fallback_used": False,
                    "fallback_reason": None,
                    "error_code": "DEEPLENS_UNAVAILABLE",
                    "error_message": "DeepLens SDK not available and strict_deeplens=True. No fallback permitted.",
                    "metadata": {
                        "backend": "deeplens",
                        "psf_source": "deeplens_parameterized",
                        "optimization_mode": "black_box",
                        "differentiable": False,
                        "native_parameter_update": False,
                        "deeplens_available": False,
                        "strict_deeplens": True,
                        "import_path": self._import_path,
                        "repo_path": self._repo_path,
                        "is_source_checkout": self._is_source_checkout,
                    },
                }
            return {
                "status": "fallback",
                "psf_cube": None,
                "psf_path": None,
                "unsupported_variables": unsupported,
                "fallback_used": True,
                "fallback_reason": "DeepLens SDK not available",
                "requested_backend": "deeplens",
                "actual_backend": "parameterized_mock",
                "deeplens_import_path": self._import_path,
                "source_checkout_used": self._is_source_checkout,
                "metadata": {
                    "backend": "deeplens",
                    "psf_source": "deeplens_parameterized",
                    "optimization_mode": "black_box",
                    "differentiable": False,
                    "native_parameter_update": False,
                    "deeplens_available": False,
                    "import_path": self._import_path,
                    "repo_path": self._repo_path,
                    "is_source_checkout": self._is_source_checkout,
                },
            }

        # Attempt DeepLens PSF generation
        try:
            config = mapping["deeplens_config"]
            psf_cube = self._try_deeplens_psf(config, output_dir)

            if psf_cube is not None:
                psf_path = output_dir / "psf_cube.npz"
                np.savez_compressed(psf_path, psf_cube=psf_cube)

                # Compute metrics
                from optiresearch.adapters.parameterized_psf import compute_psf_metrics
                metrics = compute_psf_metrics(psf_cube)
                (output_dir / "optical_metrics.json").write_text(
                    json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
                )

                manifest = {
                    "backend": "deeplens",
                    "psf_source": "deeplens_parameterized",
                    "optimization_mode": "black_box",
                    "differentiable": False,
                    "native_parameter_update": False,
                    "unsupported_variables": [u["variable"] for u in unsupported],
                    "deeplens_available": True,
                    "psf_generated": True,
                    "psf_cube_shape": list(psf_cube.shape),
                    "parameter_mapping": mapping["parameter_mapping"],
                }
                (output_dir / "psf_generation_manifest.json").write_text(
                    json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
                )

                status = "partial" if has_unsupported else "succeeded"
                return {
                    "status": status,
                    "psf_cube": psf_cube,
                    "psf_path": str(psf_path),
                    "unsupported_variables": unsupported,
                    "fallback_used": False,
                    "fallback_reason": None,
                    "psf_metrics": metrics,
                    "metadata": manifest,
                }

        except Exception as exc:
            pass  # fall through to fallback

        if self._strict:
            return {
                "status": "unavailable",
                "psf_cube": None,
                "psf_path": None,
                "unsupported_variables": unsupported,
                "fallback_used": False,
                "error_code": "DEEPLENS_PSF_FAILED",
                "error_message": "DeepLens PSF generation failed with strict_deeplens=True.",
                "metadata": {
                    "deeplens_available": self.deeplens_available,
                    "strict_deeplens": True,
                    "import_path": self._import_path,
                    "is_source_checkout": self._is_source_checkout,
                },
            }
        return {
            "status": "fallback",
            "psf_cube": None,
            "psf_path": None,
            "unsupported_variables": unsupported,
            "fallback_used": True,
            "fallback_reason": "DeepLens PSF generation failed",
            "requested_backend": "deeplens",
            "actual_backend": "parameterized_mock",
            "deeplens_import_path": self._import_path,
            "source_checkout_used": self._is_source_checkout,
            "metadata": {
                "backend": "deeplens",
                "psf_source": "deeplens_parameterized",
                "optimization_mode": "black_box",
                "differentiable": False,
                "native_parameter_update": False,
                "deeplens_available": True,
                "psf_generated": False,
                "import_path": self._import_path,
                "is_source_checkout": self._is_source_checkout,
            },
        }

    def _try_deeplens_psf(self, config: dict[str, Any], output_dir: Path) -> Optional[np.ndarray]:
        """Try to generate PSF via DeepLens ParaxialLens."""
        lens_cls = getattr(self._deeplens, "ParaxialLens", None)
        if lens_cls is None:
            return None

        optical = config.get("optical_parameters", {})
        psf_size = int(config.get("psf_size", 32))
        wavelength_count = int(config.get("wavelength_count", 31))
        depths = config.get("depths_mm", [0.0, 5.0, 10.0, 15.0, 20.0])
        chroma = float(config.get("chromatic_shift", 0.3))

        lens = lens_cls(
            foclen=float(optical.get("focal_length", 50.0)),
            fnum=float(optical.get("f_number", 2.8)),
            sensor_size=(8.0, 8.0),
            sensor_res=(max(psf_size * 64, 512), max(psf_size * 64, 512)),
            device="cpu",
        )

        psfs = []
        for d_idx, depth in enumerate(depths):
            raw_psf = self._call_paraxial_psf(lens, depth, psf_size)
            psf_2d = self._psf_to_numpy(raw_psf, psf_size)

            # Apply chromatic shift: subtle depth offsets per wavelength
            band_psfs = []
            for b in range(wavelength_count):
                wave_factor = -1.0 + 2.0 * b / max(wavelength_count - 1, 1)
                offset = chroma * wave_factor * 0.02
                shifted = np.roll(psf_2d, int(offset * psf_size), axis=0)
                shifted = np.roll(shifted, int(offset * psf_size * 0.5), axis=1)
                band_psfs.append(shifted)
            psfs.append(np.stack(band_psfs, axis=0))

        cube = np.stack(psfs, axis=0).astype(np.float32)

        # Normalize
        for d in range(cube.shape[0]):
            for b in range(cube.shape[1]):
                s = cube[d, b].sum()
                if s > 1e-12:
                    cube[d, b] /= s

        return cube

    def _call_paraxial_psf(self, lens: Any, depth: float, psf_size: int) -> Any:
        psf_fn = getattr(lens, "psf", None)
        if callable(psf_fn):
            import torch
            # ParaxialLens.psf expects points [N, 3] in (x, y, z) order
            # z is the depth axis. On-axis PSF: x=0, y=0, z=depth
            points_tensor = torch.tensor(
                [[0.0, 0.0, float(depth)]],
                device=getattr(lens, "device", "cpu"),
            )
            raw = psf_fn(points=points_tensor, ks=psf_size)
            if isinstance(raw, (list, tuple)):
                return raw[0] if len(raw) > 0 else raw
            return raw
        return None

    def _psf_to_numpy(self, raw_psf: Any, psf_size: int) -> np.ndarray:
        if raw_psf is None:
            return np.ones((psf_size, psf_size), dtype=np.float32) / (psf_size ** 2)
        if hasattr(raw_psf, "numpy"):
            arr = raw_psf.numpy()
        elif hasattr(raw_psf, "detach"):
            arr = raw_psf.detach().cpu().numpy()
        else:
            arr = np.asarray(raw_psf, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[0]
        if arr.shape != (psf_size, psf_size):
            from PIL import Image
            img = Image.fromarray((arr * 255).astype(np.uint8))
            arr = np.asarray(img.resize((psf_size, psf_size), Image.LANCZOS), dtype=np.float32)
            arr = arr / (arr.sum() + 1e-12)
        return arr.astype(np.float32)

    def _probe_capabilities(self) -> dict[str, Any]:
        try:
            from optiresearch.adapters.deeplens import DeepLensAdapter
            env = DeepLensAdapter().validate_environment()
            return {
                "deeplens_available": env.get("available", False),
                "capabilities": env.get("capabilities", []),
            }
        except Exception:
            return {"deeplens_available": False, "capabilities": []}
