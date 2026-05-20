"""Synthetic HSI dataset generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np

from optiresearch.schemas.hsi import HSIDatasetSpec


class SyntheticHSIDataset:
    def __init__(self, spec: HSIDatasetSpec, seed: int = 42) -> None:
        self.spec = spec
        self.seed = seed
        self._pattern = getattr(spec, "spectral_pattern_type", "smooth_low_rank")
        self._material_count = getattr(spec, "material_count", 6)
        self._depth_aware = getattr(spec, "depth_aware", True)

    def generate_split(self, split: Literal["train", "val", "test"]) -> dict:
        size_map = {"train": self.spec.train_size, "val": self.spec.val_size, "test": self.spec.test_size}
        size = size_map[split]
        offset = {"train": 0, "val": 1000, "test": 2000}[split]
        rng = np.random.default_rng(self.seed + offset)
        cubes = []
        depths = []
        for idx in range(size):
            depth = idx % 9
            cube = self._make_cube(rng, idx, depth)
            cubes.append(cube)
            depths.append(depth)
        return {"hsi": np.asarray(cubes, dtype=np.float32), "depth_indices": np.asarray(depths, dtype=np.int64)}

    def save(self, output_dir: Path) -> dict:
        output_dir.mkdir(parents=True, exist_ok=True)
        for split in ("train", "val", "test"):
            payload = self.generate_split(split)
            np.savez_compressed(output_dir / f"{split}.npz", **payload)
        manifest = self.spec.model_dump(mode="json")
        (output_dir / "dataset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
        return manifest

    def _make_cube(self, rng: np.random.Generator, sample_idx: int, depth: int) -> np.ndarray:
        if self._pattern == "smooth_low_rank":
            return self._make_smooth_low_rank_cube(rng, sample_idx, depth)
        if self._pattern == "mixed_materials":
            return self._make_mixed_materials_cube(rng, sample_idx, depth)
        if self._pattern == "sparse_peaks":
            return self._make_sparse_peaks_cube(rng, sample_idx, depth)
        if self._pattern == "edge_spectral_contrast":
            return self._make_edge_spectral_contrast_cube(rng, sample_idx, depth)
        return self._make_smooth_low_rank_cube(rng, sample_idx, depth)

    def _make_smooth_low_rank_cube(self, rng: np.random.Generator, sample_idx: int, depth: int) -> np.ndarray:
        bands = self.spec.spectral_bands
        height = self.spec.height
        width = self.spec.width
        yy, xx = np.meshgrid(np.linspace(-1.0, 1.0, height), np.linspace(-1.0, 1.0, width), indexing="ij")
        spatial = (
            0.45
            + 0.25 * np.sin((sample_idx + 1) * np.pi * xx)
            + 0.2 * np.cos((depth + 1) * np.pi * yy / 5.0)
            + 0.1 * np.exp(-((xx - 0.2) ** 2 + (yy + 0.1) ** 2) / 0.15)
        )
        spectral_axis = np.linspace(0.0, 1.0, bands)
        center = 0.25 + 0.5 * ((sample_idx % 5) / 4.0)
        spectrum = 0.35 + 0.45 * np.exp(-((spectral_axis - center) ** 2) / 0.04)
        spectrum += 0.12 * np.exp(-((spectral_axis - 0.75) ** 2) / 0.02)
        cube = spectrum[:, None, None] * spatial[None, :, :]
        cube += rng.normal(0.0, 0.005, size=cube.shape)
        cube = np.clip(cube, 0.0, None)
        max_value = max(float(cube.max()), 1e-8)
        return (cube / max_value).astype(np.float32)

    def _make_mixed_materials_cube(self, rng: np.random.Generator, sample_idx: int, depth: int) -> np.ndarray:
        bands = self.spec.spectral_bands
        height = self.spec.height
        width = self.spec.width
        K = self._material_count
        band_positions = np.linspace(0.0, 1.0, bands)
        material_seed = rng.integers(0, 2**31, size=K)
        spectra = np.zeros((K, bands), dtype=np.float64)
        for k in range(K):
            mat_rng = np.random.default_rng(int(material_seed[k]) + sample_idx + depth * 100)
            n_peaks = int(mat_rng.integers(1, 4))
            for _ in range(n_peaks):
                peak_center = float(mat_rng.random())
                peak_width = 0.03 + 0.07 * float(mat_rng.random())
                peak_amp = 0.3 + 0.7 * float(mat_rng.random())
                spectra[k] += peak_amp * np.exp(-((band_positions - peak_center) ** 2) / (2 * peak_width**2))
            spectra[k] = spectra[k] / max(float(spectra[k].max()), 1e-8)
        spatial_seed = rng.integers(0, 2**31, size=K)
        abundances = np.zeros((K, height, width), dtype=np.float64)
        yy, xx = np.meshgrid(np.linspace(-1.0, 1.0, height), np.linspace(-1.0, 1.0, width), indexing="ij")
        for k in range(K):
            spat_rng = np.random.default_rng(int(spatial_seed[k]) + sample_idx + depth * 100)
            cx = -0.6 + 1.2 * float(spat_rng.random())
            cy = -0.6 + 1.2 * float(spat_rng.random())
            sx = 0.15 + 0.35 * float(spat_rng.random())
            sy = 0.15 + 0.35 * float(spat_rng.random())
            abundances[k] = np.exp(-((xx - cx) ** 2 / (2 * sx**2) + (yy - cy) ** 2 / (2 * sy**2)))
        abundances_sum = np.sum(abundances, axis=0, keepdims=True)
        abundances = abundances / np.maximum(abundances_sum, 1e-8)
        if self._depth_aware:
            depth_factor = 0.7 + 0.3 * (depth / 8.0)
            depth_shift = int(depth) % K
            abundances = np.roll(abundances, depth_shift, axis=0)
            abundances *= depth_factor
        cube = np.zeros((bands, height, width), dtype=np.float64)
        for k in range(K):
            cube += spectra[k][:, None, None] * abundances[k][None, :, :]
        cube += rng.normal(0.0, 0.003, size=cube.shape)
        cube = np.clip(cube, 0.0, None)
        max_value = max(float(cube.max()), 1e-8)
        return (cube / max_value).astype(np.float32)

    def _make_sparse_peaks_cube(self, rng: np.random.Generator, sample_idx: int, depth: int) -> np.ndarray:
        bands = self.spec.spectral_bands
        height = self.spec.height
        width = self.spec.width
        band_positions = np.linspace(0.0, 1.0, bands)
        n_peaks = int(rng.integers(3, 8))
        cube = np.zeros((bands, height, width), dtype=np.float64)
        yy, xx = np.meshgrid(np.linspace(-1.0, 1.0, height), np.linspace(-1.0, 1.0, width), indexing="ij")
        for _ in range(n_peaks):
            peak_band = float(rng.random())
            peak_spectrum = np.exp(-((band_positions - peak_band) ** 2) / (2 * 0.015**2))
            cx = -0.7 + 1.4 * float(rng.random())
            cy = -0.7 + 1.4 * float(rng.random())
            sx = 0.08 + 0.25 * float(rng.random())
            sy = 0.08 + 0.25 * float(rng.random())
            spatial_patch = np.exp(-((xx - cx) ** 2 / (2 * sx**2) + (yy - cy) ** 2 / (2 * sy**2)))
            spatial_patch *= 0.3 + 0.7 * float(rng.random())
            cube += peak_spectrum[:, None, None] * spatial_patch[None, :, :]
        if self._depth_aware:
            depth_blur = max(1, int(1 + 0.5 * depth))
            for b in range(bands):
                cube[b] = _box_blur(cube[b], depth_blur)
        cube += rng.normal(0.0, 0.002, size=cube.shape)
        cube = np.clip(cube, 0.0, None)
        max_value = max(float(cube.max()), 1e-8)
        return (cube / max_value).astype(np.float32)

    def _make_edge_spectral_contrast_cube(self, rng: np.random.Generator, sample_idx: int, depth: int) -> np.ndarray:
        bands = self.spec.spectral_bands
        height = self.spec.height
        width = self.spec.width
        yy, xx = np.meshgrid(np.linspace(-1.0, 1.0, height), np.linspace(-1.0, 1.0, width), indexing="ij")
        edge_mask = np.abs(xx) > 0.4
        interior_mask = ~edge_mask
        band_positions = np.linspace(0.0, 1.0, bands)
        edge_center = 0.2 + 0.3 * float(rng.random())
        interior_center = 0.5 + 0.4 * float(rng.random())
        edge_spectrum = np.exp(-((band_positions - edge_center) ** 2) / (2 * 0.05**2))
        interior_spectrum = np.exp(-((band_positions - interior_center) ** 2) / (2 * 0.06**2))
        interior_spectrum += 0.3 * np.exp(-((band_positions - (interior_center + 0.15)) ** 2) / (2 * 0.03**2))
        cube = np.zeros((bands, height, width), dtype=np.float64)
        for b in range(bands):
            cube[b][interior_mask] = interior_spectrum[b] * 0.8
            cube[b][edge_mask] = edge_spectrum[b] * 0.6
        if self._depth_aware:
            depth_contrast = 0.5 + 0.5 * (depth / 8.0)
            cube *= (0.7 + 0.3 * depth_contrast)
        cube += rng.normal(0.0, 0.004, size=cube.shape)
        cube = np.clip(cube, 0.0, None)
        max_value = max(float(cube.max()), 1e-8)
        return (cube / max_value).astype(np.float32)


def _box_blur(image: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 1:
        return image
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float64) / (kernel_size**2)
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")
    out = np.zeros_like(image, dtype=np.float64)
    flipped = kernel[::-1, ::-1]
    ih, iw = image.shape
    for y in range(ih):
        for x in range(iw):
            out[y, x] = float(np.sum(padded[y:y + kh, x:x + kw] * flipped))
    return out
