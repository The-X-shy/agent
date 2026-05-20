"""Parameterized PSF generator for optical-HSI co-design.

Generates PSF cubes from optical variable parameters, enabling
agent-driven or gradient-free optimization of optical designs.

Optical variables:
  - phase_mask_strength: wavefront modulation amplitude
  - doe_grating_period: diffractive element grating spacing
  - surface_curvature: lens curvature / focus control
  - chromatic_shift: wavelength-dependent PSF shift
  - depth_variation: depth-dependent PSF variation
"""

from __future__ import annotations

from typing import Any

import numpy as np


def generate_parameterized_psf(
    optical_vars: dict[str, float],
    depth_planes: int = 5,
    wavelength_bands: int = 31,
    psf_size: int = 32,
    seed: int = 42,
    encoder_type: str = "controlled_chromatic_edof",
) -> np.ndarray:
    """Generate a PSF cube [D, B, H, W] from optical variables.

    The PSF cube varies predictably with each optical variable:
    - phase_mask_strength: controls spatial modulation of wavefront → PSF shape complexity
    - doe_grating_period: controls diffraction → PSF side-lobe spacing
    - surface_curvature: controls focus → PSF core width
    - chromatic_shift: controls wavelength-dependent centroid shift
    - depth_variation: controls depth-dependent PSF spread variation

    Args:
        optical_vars: Dict with keys matching optical variable names.
        depth_planes: Number of depth planes (D).
        wavelength_bands: Number of wavelength bands (B).
        psf_size: Spatial size (H, W) of each PSF slice.
        seed: Random seed for reproducibility.
        encoder_type: Encoder type for additional shaping.

    Returns:
        np.ndarray of shape [D, B, H, W] with normalized PSF slices.
    """
    rng = np.random.default_rng(seed)
    pm = float(optical_vars.get("phase_mask_strength", 0.5))
    doe = float(optical_vars.get("doe_grating_period", 1.0))
    curv = float(optical_vars.get("surface_curvature", 0.5))
    chroma = float(optical_vars.get("chromatic_shift", 0.3))
    depth_var = float(optical_vars.get("depth_variation", 0.5))

    cube = np.zeros((depth_planes, wavelength_bands, psf_size, psf_size), dtype=np.float64)
    y, x = np.meshgrid(
        np.linspace(-1, 1, psf_size),
        np.linspace(-1, 1, psf_size),
        indexing="ij",
    )

    for d in range(depth_planes):
        depth_factor = -1.0 + 2.0 * d / max(depth_planes - 1, 1)
        for b in range(wavelength_bands):
            wave_factor = -1.0 + 2.0 * b / max(wavelength_bands - 1, 1)

            # Core sigma: surface curvature controls base spread
            base_sigma = 0.12 + 0.08 * (1.0 - curv)  # high curvature → tight focus
            depth_term = depth_var * depth_factor * 0.06
            wave_term = chroma * wave_factor * 0.04
            sigma_x = base_sigma + depth_term + wave_term
            sigma_y = base_sigma + depth_term - wave_term * 0.5

            # Phase mask modulation: adds structured spatial variation
            if pm > 0.01:
                modulation = 1.0 + pm * 0.3 * np.sin(
                    2.0 * np.pi * (x + y) / (0.5 + doe * 2.0)
                )
                sigma_x = sigma_x * modulation
                sigma_y = sigma_y * modulation

            # DOE grating: shifts PSF centroid periodically
            cx = doe * 0.15 * np.sin(2.0 * np.pi * wave_factor / doe) if doe > 0.1 else 0.0
            cy = doe * 0.10 * np.cos(2.0 * np.pi * depth_factor / doe) if doe > 0.1 else 0.0

            # Build 2D Gaussian PSF
            xc = x - cx
            yc = y - cy
            gauss = np.exp(-0.5 * (xc**2 / (sigma_x**2 + 1e-8) + yc**2 / (sigma_y**2 + 1e-8)))
            gauss = gauss / (gauss.sum() + 1e-12)

            # Tiny noise for realism
            gauss = gauss + rng.normal(0, 0.0005, (psf_size, psf_size))
            gauss = np.maximum(gauss, 0.0)
            gauss = gauss / (gauss.sum() + 1e-12)

            cube[d, b] = gauss

    # Apply encoder-type specific post-processing
    cube = _apply_encoder_profile(cube, encoder_type, rng)

    return cube


def _apply_encoder_profile(cube: np.ndarray, encoder_type: str, rng: np.random.Generator) -> np.ndarray:
    """Apply encoder-specific PSF characteristics."""
    D, B, H, W = cube.shape

    if encoder_type == "conventional":
        # Weak depth variation, limited spectral variation
        for d in range(D):
            factor = 1.0 + 0.15 * (d - D // 2) / max(D // 2, 1)
            cube[d] *= factor
    elif encoder_type == "achromatic":
        # Strongly smoothed wavelength response (broadband)
        for b in range(B):
            factor = 1.0 - 0.3 * abs(b - B // 2) / max(B // 2, 1)
            cube[:, b] *= factor
    elif encoder_type == "edof":
        # Extended depth of field: strong depth smoothing
        for d in range(D):
            # Gaussian-like depth weighting
            factor = 0.6 + 0.4 * np.exp(-0.5 * ((d - D // 2) / (D / 4.0)) ** 2)
            cube[d] *= factor
    elif encoder_type == "chromatic_coded":
        # Strong spectral coding with spatial shifts
        for b in range(B):
            phase = 2.0 * np.pi * b / max(B - 1, 1)
            shift_x = int(H * 0.3 * np.sin(phase))
            shift_y = int(H * 0.15 * np.cos(phase))
            cube[:, b] = np.roll(cube[:, b], shift_x, axis=-1)
            cube[:, b] = np.roll(cube[:, b], shift_y, axis=-2)
    elif encoder_type == "controlled_chromatic_edof":
        # Combined: depth smoothing + spectral coding with spatial shifts
        for d in range(D):
            factor = 0.6 + 0.4 * np.exp(-0.5 * ((d - D // 2) / (D / 4.0)) ** 2)
            cube[d] *= factor
        for b in range(B):
            phase = 2.0 * np.pi * b / max(B - 1, 1)
            shift_x = int(H * 0.3 * np.sin(phase))
            shift_y = int(H * 0.15 * np.cos(phase))
            cube[:, b] = np.roll(cube[:, b], shift_x, axis=-1)
            cube[:, b] = np.roll(cube[:, b], shift_y, axis=-2)

    # Per-slice normalization (energy conservation per PSF slice)
    for d in range(D):
        for b in range(B):
            s = cube[d, b].sum()
            if s > 1e-12:
                cube[d, b] /= s

    return cube


def compute_psf_metrics(cube: np.ndarray) -> dict[str, Any]:
    """Compute optical metrics from a PSF cube using structural measures."""
    D, B, H, W = cube.shape

    # Centroid positions per slice (sensitive to spatial shifts)
    centroids_y = []
    centroids_x = []
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

    for d in range(D):
        for b in range(B):
            sl = cube[d, b]
            total = sl.sum()
            if total > 1e-12:
                cy = float((yy * sl).sum() / total)
                cx = float((xx * sl).sum() / total)
                centroids_y.append(cy)
                centroids_x.append(cx)

    centroids_y = np.array(centroids_y)
    centroids_x = np.array(centroids_x)

    # Depth stability: std of centroids across depth planes (per band averaged)
    depth_centroid_std = 0.0
    for b in range(B):
        b_cy = np.array([centroids_y[d * B + b] for d in range(D)])
        depth_centroid_std += float(np.std(b_cy))
    depth_centroid_std /= B
    depth_stability = 1.0 - min(1.0, depth_centroid_std / (H / 4.0))

    # Spectral separability: std of centroids across bands (per depth averaged)
    spectral_centroid_std = 0.0
    for d in range(D):
        d_cx = np.array([centroids_x[d * B + b] for b in range(B)])
        spectral_centroid_std += float(np.std(d_cx))
    spectral_centroid_std /= D
    spectral_sep = min(1.0, spectral_centroid_std / (W / 8.0))

    # Coding strength: spectral_sep * (1 - depth_stability)
    coding_strength = spectral_sep * (1.0 - depth_stability)

    # Per-slice spatial spread variation
    spreads = []
    for d in range(D):
        for b in range(B):
            sl = cube[d, b]
            total = sl.sum()
            if total > 1e-12:
                cy = (yy * sl).sum() / total
                cx = (xx * sl).sum() / total
                var_y = ((yy - cy) ** 2 * sl).sum() / total
                var_x = ((xx - cx) ** 2 * sl).sum() / total
                spreads.append(float(np.sqrt(var_y + var_x)))
    spreads_arr = np.array(spreads)
    band_condition = float(np.std(spreads_arr) / (np.mean(spreads_arr) + 1e-8))

    return {
        "depth_stability_score": round(depth_stability, 6),
        "spectral_separability_score": round(spectral_sep, 6),
        "coding_strength": round(coding_strength, 6),
        "band_condition_score": round(band_condition, 6),
        "psf_cube_shape": list(cube.shape),
        "psf_mean_intensity": round(float(cube.mean()), 6),
        "psf_max_intensity": round(float(cube.max()), 6),
    }


def optical_vars_to_dict(variables: list[Any]) -> dict[str, float]:
    """Convert OpticalVariable list to current-value dict."""
    result: dict[str, float] = {}
    for v in variables:
        if hasattr(v, "name") and hasattr(v, "current_value"):
            result[v.name] = float(v.current_value)
        elif isinstance(v, dict):
            result[v.get("name", "")] = float(v.get("current_value", 0.5))
    return result
