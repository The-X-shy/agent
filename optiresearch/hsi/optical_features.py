"""Optical feature extraction from PSF cubes for HSI reconstruction."""

from __future__ import annotations

import numpy as np


class OpticalFeatureExtractor:
    """Extract band-dependent and encoder-characterizing features from a PSF cube.

    Input: psf_cube of shape (depth_planes, wavelength_bands, psf_height, psf_width).
    Output: dict of scalar and per-band features usable by forward models and reconstructors.
    """

    def extract(self, psf_cube: np.ndarray) -> dict:
        cube = np.asarray(psf_cube, dtype=np.float64)
        if cube.ndim != 4:
            raise ValueError(f"Expected 4D PSF cube (D, B, H, W), got shape {cube.shape}")
        D, B, H, W = cube.shape

        centroids_x, centroids_y = self._band_centroids(cube)
        spread = self._band_spread(cube, centroids_x, centroids_y)
        high_freq = self._band_high_freq_energy(cube)
        depth_stability = self._depth_stability_score(cube)
        spectral_sep = self._spectral_separability_score(cube, centroids_x, centroids_y)
        coding_strength = round(float(spectral_sep * (1.0 - depth_stability)), 6)
        band_condition = self._band_condition_score(cube, spread)

        return {
            "band_spread": spread.astype(np.float32),
            "band_centroid_x": centroids_x.astype(np.float32),
            "band_centroid_y": centroids_y.astype(np.float32),
            "band_high_freq_energy": high_freq.astype(np.float32),
            "depth_stability_score": depth_stability,
            "spectral_separability_score": spectral_sep,
            "coding_strength": coding_strength,
            "band_condition_score": band_condition,
            "depth_planes": D,
            "wavelength_bands": B,
        }

    def _band_centroids(self, cube: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        D, B, H, W = cube.shape
        yy = np.arange(H, dtype=np.float64).reshape(H, 1)
        xx = np.arange(W, dtype=np.float64).reshape(1, W)
        centroids_x = np.zeros(B, dtype=np.float64)
        centroids_y = np.zeros(B, dtype=np.float64)
        center_depth = D // 2
        for b in range(B):
            psf = cube[center_depth, b]
            total = max(float(psf.sum()), 1e-12)
            centroids_x[b] = float(np.sum(psf * xx) / total)
            centroids_y[b] = float(np.sum(psf * yy) / total)
        return centroids_x, centroids_y

    def _band_spread(self, cube: np.ndarray, cx: np.ndarray, cy: np.ndarray) -> np.ndarray:
        D, B, H, W = cube.shape
        yy = np.arange(H, dtype=np.float64).reshape(H, 1)
        xx = np.arange(W, dtype=np.float64).reshape(1, W)
        spread = np.zeros(B, dtype=np.float64)
        center_depth = D // 2
        for b in range(B):
            psf = cube[center_depth, b]
            total = max(float(psf.sum()), 1e-12)
            var_x = float(np.sum(psf * (xx - cx[b]) ** 2) / total)
            var_y = float(np.sum(psf * (yy - cy[b]) ** 2) / total)
            spread[b] = np.sqrt(var_x + var_y)
        return spread

    def _band_high_freq_energy(self, cube: np.ndarray) -> np.ndarray:
        D, B, H, W = cube.shape
        high_freq = np.zeros(B, dtype=np.float64)
        center_depth = D // 2
        fy = np.fft.fftfreq(H, d=1.0 / H)
        fx = np.fft.rfftfreq(W, d=1.0 / W)
        fy2, fx2 = np.meshgrid(np.abs(fy), np.abs(fx), indexing="ij")
        radius = np.sqrt(fy2**2 + fx2**2)
        high_mask = radius > (max(H, W) * 0.15)
        for b in range(B):
            psf = cube[center_depth, b]
            fft = np.abs(np.fft.rfft2(psf))
            total = max(float(fft.sum()), 1e-12)
            high_freq[b] = float(fft[high_mask].sum()) / total
        return high_freq

    def _depth_stability_score(self, cube: np.ndarray) -> float:
        D, B, H, W = cube.shape
        center_band = B // 2
        center_depth = D // 2
        ref = cube[center_depth, center_band].ravel()
        ref_norm = max(float(np.linalg.norm(ref)), 1e-12)
        similarities = []
        for d in range(D):
            vec = cube[d, center_band].ravel()
            vec_norm = max(float(np.linalg.norm(vec)), 1e-12)
            sim = float(np.dot(ref, vec) / (ref_norm * vec_norm))
            similarities.append(sim)
        return round(float(np.mean(similarities)), 6)

    def _spectral_separability_score(self, cube: np.ndarray, cx: np.ndarray, cy: np.ndarray) -> float:
        D, B, H, W = cube.shape
        center_depth = D // 2
        band_profiles = cube[center_depth].reshape(B, -1)
        l1_diff = 0.0
        count = 0
        for i in range(B):
            for j in range(i + 1, B):
                l1_diff += float(np.mean(np.abs(band_profiles[i] - band_profiles[j])))
                count += 1
        mean_l1 = l1_diff / max(count, 1)
        centroid_span = float(np.sqrt((cx.max() - cx.min()) ** 2 + (cy.max() - cy.min()) ** 2))
        sep_score = float(np.clip(mean_l1 * 2.0 + centroid_span * 0.5, 0.0, 1.0))
        return round(sep_score, 6)

    def _band_condition_score(self, cube: np.ndarray, spread: np.ndarray) -> float:
        spread_range = float(spread.max() - spread.min())
        spread_mean = max(float(spread.mean()), 1e-12)
        return round(float(np.clip(spread_range / spread_mean, 0.0, 1.0)), 6)
