"""Deterministic mock DeepLens adapter for MVP testing."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from optiresearch.adapters.base import AdapterArtifact, AdapterMetricBundle, AdapterRunResult
from optiresearch.adapters.metrics import clamp01
from optiresearch.memory.schemas import compute_file_sha256
from optiresearch.runtime.backend_metadata import backend_metadata, enrich_backend_metrics
from optiresearch.schemas.experiment import ExperimentSpec, OpticalSpec, SweepSpec, validate_experiment_spec_version


class MockDeepLensAdapter:
    """Generate stable PSF/MTF artifacts without requiring real DeepLens."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def simulate_psf_cube(
        self,
        spec: dict[str, Any] | OpticalSpec | ExperimentSpec,
        sweep: dict[str, Any] | SweepSpec | None,
        output_dir: Path,
    ) -> AdapterRunResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        if isinstance(spec, ExperimentSpec):
            validate_experiment_spec_version(spec)
        spec_dict, sweep_dict = self._normalize_specs(spec, sweep)
        depth_planes = int(sweep_dict.get("depth_planes", spec_dict.get("depth_planes", 9)))
        wavelength_bands = int(sweep_dict.get("wavelength_bands", spec_dict.get("wavelength_bands", 31)))
        psf_size = int(spec_dict.get("psf_size", 32))
        encoder_type = str(spec_dict.get("encoder_type", "mock"))
        cube = self._make_psf_cube(depth_planes, wavelength_bands, psf_size, encoder_type)
        metrics = enrich_backend_metrics(
            self._metrics(cube, depth_planes, wavelength_bands, str(spec_dict.get("encoder_type", "mock"))),
            "mock_deeplens",
        )
        paths = self._write_outputs(output_dir, cube, metrics, spec_dict, sweep_dict)
        return AdapterRunResult(
            status="succeeded",
            artifacts=[str(path) for path in paths],
            artifact_refs=[self._adapter_artifact(path) for path in paths],
            metric_bundle=AdapterMetricBundle(
                metrics=metrics,
                primary_metric="psf_depth_similarity",
                thresholds={"psf_depth_similarity": 0.8, "spectral_separability": 0.3},
                metadata=backend_metadata("mock_deeplens", {"seed": self.seed}),
            ),
            logs=[f"Generated mock PSF cube with seed={self.seed}"],
            errors=[],
            metadata={
                **backend_metadata("mock_deeplens"),
                "encoder_type": metrics.get("encoder_type"),
                "schema_version": spec_dict.get("schema_version", "0.1"),
            },
        )

    def _normalize_specs(
        self,
        spec: dict[str, Any] | OpticalSpec | ExperimentSpec,
        sweep: dict[str, Any] | SweepSpec | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if isinstance(spec, ExperimentSpec):
            return self._optical_dict(spec.optical_spec), self._sweep_dict(spec.sweep_spec)
        if isinstance(spec, OpticalSpec):
            return self._optical_dict(spec), self._sweep_dict(sweep)
        return dict(spec), self._sweep_dict(sweep)

    def _optical_dict(self, spec: OpticalSpec) -> dict[str, Any]:
        payload = spec.model_dump(mode="json")
        payload["depth_planes"] = spec.depth_planes
        payload["wavelength_bands"] = spec.wavelength_bands
        return payload

    def _sweep_dict(self, sweep: dict[str, Any] | SweepSpec | None) -> dict[str, Any]:
        if isinstance(sweep, SweepSpec):
            return {
                "depth_planes": len(sweep.depths_mm),
                "wavelength_bands": len(sweep.wavelengths_nm),
                **sweep.model_dump(mode="json"),
            }
        return dict(sweep or {})

    def _make_psf_cube(self, depth_planes: int, wavelength_bands: int, psf_size: int, encoder_type: str = "mock") -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        axis = np.linspace(-1.0, 1.0, psf_size)
        xx, yy = np.meshgrid(axis, axis)
        cube = np.zeros((depth_planes, wavelength_bands, psf_size, psf_size), dtype=np.float32)
        center_depth = (depth_planes - 1) / 2.0
        center_wave = (wavelength_bands - 1) / 2.0
        for depth in range(depth_planes):
            for band in range(wavelength_bands):
                depth_term = abs(depth - center_depth) / max(center_depth, 1.0)
                wave_term = (band - center_wave) / max(center_wave, 1.0)
                sigma_x = 0.18 + 0.025 * depth_term + 0.012 * abs(wave_term)
                sigma_y = 0.18 + 0.018 * depth_term + 0.018 * abs(wave_term)
                offset = 0.04 * wave_term
                psf = np.exp(-(((xx - offset) ** 2) / (2 * sigma_x**2) + (yy**2) / (2 * sigma_y**2)))
                psf += rng.normal(0.0, 0.0005, size=psf.shape)
                psf = np.clip(psf, 0.0, None)
                psf = psf / max(float(psf.sum()), 1e-8)
                cube[depth, band] = psf
        cube = self._apply_encoder_transform(cube, encoder_type, depth_planes, wavelength_bands)
        return cube

    def _apply_encoder_transform(self, cube: np.ndarray, encoder_type: str, depth_planes: int, wavelength_bands: int) -> np.ndarray:
        if encoder_type == "achromatic":
            return self._smooth_wavelength_response(cube, strength=0.88)
        if encoder_type == "edof":
            return self._smooth_depth_response(cube, strength=0.78)
        if encoder_type == "chromatic_coded":
            return self._add_spectral_code(cube, strength=0.92)
        if encoder_type == "controlled_chromatic_edof":
            smoothed = self._smooth_depth_response(cube, strength=0.65)
            return self._add_spectral_code(smoothed, strength=0.75)
        return cube

    @staticmethod
    def _smooth_wavelength_response(cube: np.ndarray, strength: float) -> np.ndarray:
        D, B, H, W = cube.shape
        for d in range(D):
            mean_psf = cube[d].mean(axis=0)
            for b in range(B):
                cube[d, b] = (1.0 - strength) * cube[d, b] + strength * mean_psf
        return cube

    @staticmethod
    def _smooth_depth_response(cube: np.ndarray, strength: float) -> np.ndarray:
        D, B, H, W = cube.shape
        for b in range(B):
            mean_psf = cube[:, b].mean(axis=0)
            for d in range(D):
                cube[d, b] = (1.0 - strength) * cube[d, b] + strength * mean_psf
        return cube

    @staticmethod
    def _add_spectral_code(cube: np.ndarray, strength: float) -> np.ndarray:
        D, B, H, W = cube.shape
        rng = np.random.default_rng(42)
        for b in range(B):
            shift_x = int(round((b / max(B - 1, 1) - 0.5) * strength * 4.0))
            shifted = np.zeros_like(cube[:, b])
            if shift_x > 0:
                shifted[:, :, shift_x:] = cube[:, b, :, :W - shift_x]
            elif shift_x < 0:
                shifted[:, :, :W + shift_x] = cube[:, b, :, -shift_x:]
            else:
                shifted = cube[:, b]
            cube[:, b] = shifted
        return cube

    def _metrics(self, cube: np.ndarray, depth_planes: int, wavelength_bands: int, encoder_type: str) -> dict[str, Any]:
        center_depth = depth_planes // 2
        center_band = wavelength_bands // 2
        depth_diff = np.mean(np.abs(cube[:, center_band] - cube[center_depth, center_band]))
        band_profiles = cube[center_depth].reshape(wavelength_bands, -1)
        spectral_var = float(np.mean(np.var(band_profiles, axis=0)) * 1000.0)
        fft_mag = np.abs(np.fft.rfft2(cube[center_depth, center_band]))
        mtf_mean = float(np.mean(fft_mag / max(float(fft_mag.max()), 1e-8)))
        profile = self._encoder_profile(encoder_type)
        return {
            "encoder_type": encoder_type,
            "depth_planes": depth_planes,
            "wavelength_bands": wavelength_bands,
            "psf_depth_similarity": profile["psf_depth_similarity"],
            "spectral_separability": profile["spectral_separability"],
            "mock_mtf_mean": profile["mock_mtf_mean"],
            "mock_energy_efficiency": profile["mock_energy_efficiency"],
            "raw_psf_depth_similarity": round(clamp01(1.0 - float(depth_diff) * 500.0), 6),
            "raw_spectral_separability": round(clamp01(spectral_var), 6),
            "raw_mock_mtf_mean": round(clamp01(mtf_mean), 6),
        }

    def _encoder_profile(self, encoder_type: str) -> dict[str, float]:
        profiles = {
            "conventional": {
                "psf_depth_similarity": 0.42,
                "spectral_separability": 0.08,
                "mock_mtf_mean": 0.72,
                "mock_energy_efficiency": 0.91,
            },
            "achromatic": {
                "psf_depth_similarity": 0.78,
                "spectral_separability": 0.06,
                "mock_mtf_mean": 0.82,
                "mock_energy_efficiency": 0.88,
            },
            "edof": {
                "psf_depth_similarity": 0.9,
                "spectral_separability": 0.14,
                "mock_mtf_mean": 0.58,
                "mock_energy_efficiency": 0.8,
            },
            "chromatic_coded": {
                "psf_depth_similarity": 0.5,
                "spectral_separability": 0.72,
                "mock_mtf_mean": 0.48,
                "mock_energy_efficiency": 0.73,
            },
            "controlled_chromatic_edof": {
                "psf_depth_similarity": 0.88,
                "spectral_separability": 0.64,
                "mock_mtf_mean": 0.66,
                "mock_energy_efficiency": 0.84,
            },
            "mock": {
                "psf_depth_similarity": 0.86,
                "spectral_separability": 0.5,
                "mock_mtf_mean": 0.62,
                "mock_energy_efficiency": 0.83,
            },
        }
        return profiles.get(encoder_type, profiles["mock"])

    def _write_outputs(
        self,
        output_dir: Path,
        cube: np.ndarray,
        metrics: dict[str, Any],
        spec: dict[str, Any],
        sweep: dict[str, Any],
    ) -> list[Path]:
        paths: list[Path] = []
        psf_path = output_dir / "psf_cube.npz"
        np.savez_compressed(psf_path, psf_cube=cube)
        paths.append(psf_path)

        mtf_path = output_dir / "mtf_curves.csv"
        with mtf_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["frequency", "mtf"])
            for frequency in np.linspace(0.0, 1.0, 25):
                writer.writerow([round(float(frequency), 6), round(float(np.exp(-2.0 * frequency)), 6)])
        paths.append(mtf_path)

        metrics_path = output_dir / "optical_metrics.json"
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
        paths.append(metrics_path)

        manifest_path = output_dir / "run_manifest.json"
        manifest_path.write_text(
            json.dumps(
                {"backend": "mock_deeplens", "seed": self.seed, "spec": spec, "sweep": sweep},
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            ),
            encoding="utf-8",
        )
        paths.append(manifest_path)

        plot_path = self._try_write_plot(output_dir, cube)
        if plot_path:
            paths.append(plot_path)
        return paths

    def _try_write_plot(self, output_dir: Path, cube: np.ndarray) -> Path | None:
        try:
            import matplotlib.pyplot as plt
        except Exception:
            return None
        center_band = cube.shape[1] // 2
        fig, axes = plt.subplots(1, min(3, cube.shape[0]), figsize=(6, 2))
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        for idx, axis in enumerate(axes):
            depth = int(idx * (cube.shape[0] - 1) / max(len(axes) - 1, 1))
            axis.imshow(cube[depth, center_band], cmap="viridis")
            axis.set_axis_off()
            axis.set_title(f"z{depth}")
        path = output_dir / "psf_grid.png"
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path

    def _adapter_artifact(self, path: Path) -> AdapterArtifact:
        return AdapterArtifact(
            path=str(path),
            artifact_type=self._artifact_type(path.name),
            mime=self._mime(path.name),
            content_hash=compute_file_sha256(path),
            metadata=backend_metadata("mock_deeplens", {"filename": path.name}),
        )

    def _artifact_type(self, filename: str) -> str:
        lower = filename.lower()
        if lower.endswith(".npz") or "psf_cube" in lower:
            return "psf_cube"
        if lower.endswith(".csv") or "mtf" in lower:
            return "mtf_curve"
        if lower.endswith(".json") and "metrics" in lower:
            return "metrics"
        if lower.endswith(".json") and "manifest" in lower:
            return "manifest"
        if lower.endswith(".png"):
            return "figure"
        return "unknown"

    def _mime(self, filename: str) -> str | None:
        lower = filename.lower()
        if lower.endswith(".json"):
            return "application/json"
        if lower.endswith(".csv"):
            return "text/csv"
        if lower.endswith(".npz"):
            return "application/octet-stream"
        if lower.endswith(".png"):
            return "image/png"
        return None
