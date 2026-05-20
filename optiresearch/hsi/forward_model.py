"""Wavelength-aware HSI forward model with optical-sensitive rendering modes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from optiresearch.hsi.optical_features import OpticalFeatureExtractor
from optiresearch.schemas.hsi import HSIForwardModelSpec


class HSIForwardModel:
    def __init__(self, spec: HSIForwardModelSpec) -> None:
        self.spec = spec
        self._mode = getattr(spec, "forward_mode", "simple_sum")
        self._normalize = getattr(spec, "normalize_measurement", True)
        self._preserve_contrast = getattr(spec, "preserve_encoder_contrast", True)

    def load_psf_cube(
        self,
        psf_cube_uri: str,
        hsi_wavelengths: list[float] | None = None,
        psf_wavelengths: list[float] | None = None,
        resample_psf_bands: bool = False,
    ) -> np.ndarray:
        path = Path(psf_cube_uri)
        payload = np.load(path)
        for key in ("psf_cube", "raw_base_psf_cube", "cube", "psf"):
            if key in payload.files:
                cube = np.asarray(payload[key], dtype=np.float32)
                break
        else:
            cube = np.asarray(payload[payload.files[0]], dtype=np.float32)
        manifest = _read_wavelength_manifest(path)
        psf_wavelengths = psf_wavelengths or manifest.get("wavelengths_nm")
        if hsi_wavelengths is not None:
            validation = validate_psf_hsi_compatibility(cube, psf_wavelengths, hsi_wavelengths, resample_psf_bands=resample_psf_bands)
            if validation["status"] == "error":
                raise ValueError(json.dumps(validation, sort_keys=True))
            if validation.get("resampled"):
                cube = align_psf_to_hsi_wavelengths(cube, psf_wavelengths, hsi_wavelengths)
        return cube

    def render_measurement(
        self,
        hsi_cube: np.ndarray,
        psf_cube: np.ndarray,
        depth_index: int = 0,
        optical_features: dict | None = None,
    ) -> np.ndarray:
        hsi = np.asarray(hsi_cube, dtype=np.float32)
        psf = np.asarray(psf_cube, dtype=np.float32)
        depth = int(np.clip(depth_index, 0, psf.shape[0] - 1))
        bands = min(hsi.shape[0], psf.shape[1])

        if self._mode == "simple_sum":
            return self._render_simple_sum(hsi, psf, depth, bands)
        if self._mode == "psf_weighted":
            return self._render_psf_weighted(hsi, psf, depth, bands)
        if self._mode == "coded_aperture_proxy":
            return self._render_coded_aperture_proxy(hsi, psf, depth, bands, optical_features)
        if self._mode == "depth_spectral_coded":
            return self._render_depth_spectral_coded(hsi, psf, depth, bands, optical_features)
        return self._render_simple_sum(hsi, psf, depth, bands)

    def render_batch(
        self,
        hsi_batch: np.ndarray,
        psf_cube: np.ndarray,
        depth_indices: Any = None,
        optical_features: dict | None = None,
    ) -> dict:
        hsi = np.asarray(hsi_batch, dtype=np.float32)
        if depth_indices is None:
            depth_indices = np.zeros(hsi.shape[0], dtype=np.int64)
        depth_indices = np.asarray(depth_indices, dtype=np.int64)
        if optical_features is None and self._mode != "simple_sum":
            optical_features = OpticalFeatureExtractor().extract(psf_cube)
        measurements = [
            self.render_measurement(hsi[idx], psf_cube, int(depth_indices[idx]), optical_features)
            for idx in range(hsi.shape[0])
        ]
        return {
            "measurements": np.asarray(measurements, dtype=np.float32),
            "targets": hsi,
            "depth_indices": depth_indices,
        }

    def _render_simple_sum(self, hsi: np.ndarray, psf: np.ndarray, depth: int, bands: int) -> np.ndarray:
        measurement = np.zeros(hsi.shape[-2:], dtype=np.float32)
        for band in range(bands):
            measurement += _conv2d_same(hsi[band], psf[depth, band])
        measurement = measurement / max(float(bands), 1.0)
        return measurement[None, :, :].astype(np.float32)

    def _render_psf_weighted(self, hsi: np.ndarray, psf: np.ndarray, depth: int, bands: int) -> np.ndarray:
        measurement = np.zeros(hsi.shape[-2:], dtype=np.float32)
        total_weight = 0.0
        for band in range(bands):
            psf_slice = psf[depth, band]
            weight = float(np.sum(psf_slice**2))
            convolved = _conv2d_same(hsi[band], psf_slice)
            measurement += convolved * weight
            total_weight += weight
        measurement = measurement / max(total_weight, 1e-8)
        return measurement[None, :, :].astype(np.float32)

    def _render_coded_aperture_proxy(
        self,
        hsi: np.ndarray,
        psf: np.ndarray,
        depth: int,
        bands: int,
        optical_features: dict | None,
    ) -> np.ndarray:
        feats = optical_features or {}
        centroids_x = np.asarray(feats.get("band_centroid_x", np.zeros(bands)), dtype=np.float64)
        centroids_y = np.asarray(feats.get("band_centroid_y", np.zeros(bands)), dtype=np.float64)
        spread = np.asarray(feats.get("band_spread", np.ones(bands) * 0.2), dtype=np.float64)
        high_freq = np.asarray(feats.get("band_high_freq_energy", np.ones(bands) * 0.5), dtype=np.float64)

        cx_ref = float(np.mean(centroids_x))
        cy_ref = float(np.mean(centroids_y))
        sp_ref = max(float(np.mean(spread)), 1e-8)
        hf_ref = max(float(np.mean(high_freq)), 1e-8)

        H, W = hsi.shape[-2:]
        measurement = np.zeros((H, W), dtype=np.float64)
        total_weight = 0.0

        for band in range(bands):
            psf_slice = psf[depth, band]
            convolved = _conv2d_same(hsi[band], psf_slice)

            centroid_weight = 1.0 + 0.5 * (abs(centroids_x[band] - cx_ref) + abs(centroids_y[band] - cy_ref))
            spread_weight = 1.0 + 0.3 * (spread[band] / sp_ref - 1.0)
            hf_weight = 1.0 + 0.4 * (high_freq[band] / hf_ref - 1.0)
            coding_weight = float(centroid_weight * spread_weight * hf_weight)

            measurement += convolved * coding_weight
            total_weight += coding_weight

        measurement = measurement / max(total_weight, 1e-8)
        if self._normalize:
            measurement = _zero_one_normalize(measurement)
        return measurement[None, :, :].astype(np.float32)

    def _render_depth_spectral_coded(
        self,
        hsi: np.ndarray,
        psf: np.ndarray,
        depth: int,
        bands: int,
        optical_features: dict | None,
    ) -> np.ndarray:
        feats = optical_features or {}
        spread = np.asarray(feats.get("band_spread", np.ones(bands) * 0.2), dtype=np.float64)
        high_freq = np.asarray(feats.get("band_high_freq_energy", np.ones(bands) * 0.5), dtype=np.float64)
        centroids_x = np.asarray(feats.get("band_centroid_x", np.zeros(bands)), dtype=np.float64)
        centroids_y = np.asarray(feats.get("band_centroid_y", np.zeros(bands)), dtype=np.float64)
        spectral_sep = float(feats.get("spectral_separability_score", 0.3))
        depth_stability = float(feats.get("depth_stability_score", 0.8))

        H, W = hsi.shape[-2:]
        measurement = np.zeros((H, W), dtype=np.float64)

        depth_factor = 0.5 + 0.5 * (depth / max(psf.shape[0] - 1, 1))
        shift_strength = spectral_sep * 6.0

        for band in range(bands):
            psf_slice = psf[depth, band]
            convolved = _conv2d_same(hsi[band], psf_slice)

            # Apply per-band spatial shift to create coded measurement
            sx = (centroids_x[band] - centroids_x.mean()) * shift_strength
            sy = (centroids_y[band] - centroids_y.mean()) * shift_strength
            shifted = _shift_2d(convolved, sx, sy)

            spread_weight = float(spread[band] / max(float(spread.mean()), 1e-8))
            hf_weight = float(high_freq[band] / max(float(high_freq.mean()), 1e-8))
            depth_weight = depth_stability * depth_factor

            combined_weight = float(spread_weight * hf_weight * depth_weight)
            measurement += shifted * combined_weight

        measurement = measurement / max(float(bands), 1.0)
        # Apply spectral degradation proportional to (1 - spectral_sep)
        # Encoders with low spectral separability lose more band-specific information
        degradation = 1.0 - spectral_sep
        if degradation > 0.01:
            measurement = measurement * (1.0 - degradation * 0.6) + np.mean(measurement) * degradation * 0.6
        if self._normalize:
            measurement = _zero_one_normalize(measurement)
        return measurement[None, :, :].astype(np.float32)

    def render_measurement_with_coding_weights(
        self,
        hsi_cube: np.ndarray,
        psf_cube: np.ndarray,
        depth_index: int = 0,
        optical_features: dict | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Render measurement and return per-band coding weights for analysis."""
        hsi = np.asarray(hsi_cube, dtype=np.float32)
        psf = np.asarray(psf_cube, dtype=np.float32)
        depth = int(np.clip(depth_index, 0, psf.shape[0] - 1))
        bands = min(hsi.shape[0], psf.shape[1])

        if optical_features is None:
            optical_features = OpticalFeatureExtractor().extract(psf_cube)

        feats = optical_features
        spread = np.asarray(feats.get("band_spread", np.ones(bands)), dtype=np.float64)
        high_freq = np.asarray(feats.get("band_high_freq_energy", np.ones(bands)), dtype=np.float64)
        spectral_sep = float(feats.get("spectral_separability_score", 0.3))
        depth_stability = float(feats.get("depth_stability_score", 0.8))

        coding_weights = np.zeros(bands, dtype=np.float64)
        for band in range(bands):
            sp_w = float(spread[band] / max(float(spread.mean()), 1e-8))
            hf_w = float(high_freq[band] / max(float(high_freq.mean()), 1e-8))
            coding_weights[band] = (0.5 + spectral_sep * 2.0) * sp_w * hf_w

        measurement = self.render_measurement(hsi_cube, psf_cube, depth_index, optical_features)
        return measurement, coding_weights.astype(np.float32)

    def save_forward_artifacts(
        self,
        output_dir: Path,
        optical_features: dict,
        coding_weights: np.ndarray | None = None,
        measurement_stats: dict | None = None,
    ) -> list[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []

        feat_path = output_dir / "optical_features.json"
        feat_path.write_text(json.dumps(_serialize_features(optical_features), ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
        paths.append(feat_path)

        if coding_weights is not None:
            cw_path = output_dir / "coding_weights.npy"
            np.save(cw_path, np.asarray(coding_weights, dtype=np.float32))
            paths.append(cw_path)

        if measurement_stats is not None:
            stats_path = output_dir / "measurement_stats.json"
            stats_path.write_text(json.dumps(measurement_stats, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
            paths.append(stats_path)

        manifest = {
            "forward_mode": self._mode,
            "normalize_measurement": self._normalize,
            "preserve_encoder_contrast": self._preserve_contrast,
            "optical_features_summary": {
                "depth_stability_score": optical_features.get("depth_stability_score"),
                "spectral_separability_score": optical_features.get("spectral_separability_score"),
                "coding_strength": optical_features.get("coding_strength"),
                "band_condition_score": optical_features.get("band_condition_score"),
            },
        }
        manifest_path = output_dir / "forward_model_manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
        paths.append(manifest_path)

        return paths


def _conv2d_same(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kernel = np.asarray(kernel, dtype=np.float32)
    kernel_sum = max(float(kernel.sum()), 1e-8)
    kernel = kernel / kernel_sum
    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")
    output = np.zeros_like(image, dtype=np.float32)
    flipped = kernel[::-1, ::-1]
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            output[y, x] = float(np.sum(padded[y : y + kh, x : x + kw] * flipped))
    return output


def _shift_2d(image: np.ndarray, sx: float, sy: float) -> np.ndarray:
    if abs(sx) < 0.3 and abs(sy) < 0.3:
        return image
    ix, iy = int(round(sx)), int(round(sy))
    if ix == 0 and iy == 0:
        return image
    result = np.zeros_like(image, dtype=np.float64)
    H, W = image.shape
    x_src_start = max(0, -ix)
    x_src_end = min(W, W - ix)
    x_dst_start = max(0, ix)
    x_dst_end = min(W, W + ix)
    y_src_start = max(0, -iy)
    y_src_end = min(H, H - iy)
    y_dst_start = max(0, iy)
    y_dst_end = min(H, H + iy)
    src_h = y_src_end - y_src_start
    src_w = x_src_end - x_src_start
    if src_h > 0 and src_w > 0:
        result[y_dst_start:y_dst_start + src_h, x_dst_start:x_dst_start + src_w] = image[y_src_start:y_src_start + src_h, x_src_start:x_src_start + src_w]
    return result


def _zero_one_normalize(arr: np.ndarray) -> np.ndarray:
    arr_min = float(arr.min())
    arr_max = float(arr.max())
    if arr_max - arr_min < 1e-8:
        return np.zeros_like(arr, dtype=np.float64)
    return (arr - arr_min) / (arr_max - arr_min)


def _serialize_features(features: dict) -> dict:
    serialized = {}
    for key, value in features.items():
        if isinstance(value, np.ndarray):
            serialized[key] = value.tolist()
        elif isinstance(value, (np.floating, np.integer)):
            serialized[key] = float(value)
        else:
            serialized[key] = value
    return serialized


def validate_psf_hsi_compatibility(
    psf_cube: np.ndarray,
    psf_wavelengths: list[float] | None,
    hsi_wavelengths: list[float] | None,
    resample_psf_bands: bool = False,
) -> dict[str, Any]:
    cube = np.asarray(psf_cube)
    if cube.ndim != 4:
        return {"status": "error", "error_code": "PSF_SHAPE_INVALID", "message": f"Expected [D,B,H,W], got {cube.shape}"}
    psf_bands = int(cube.shape[1])
    hsi_bands = len(hsi_wavelengths) if hsi_wavelengths is not None else int(getattr(hsi_wavelengths, "shape", [psf_bands])[0])
    if hsi_wavelengths is None or psf_bands == hsi_bands:
        return {"status": "valid", "psf_bands": psf_bands, "hsi_bands": hsi_bands, "resampled": False}
    if not resample_psf_bands:
        return {
            "status": "error",
            "error_code": "BAND_MISMATCH",
            "message": f"PSF bands ({psf_bands}) do not match HSI bands ({hsi_bands}).",
            "psf_bands": psf_bands,
            "hsi_bands": hsi_bands,
        }
    if psf_wavelengths is None or len(psf_wavelengths) != psf_bands:
        return {
            "status": "error",
            "error_code": "PSF_WAVELENGTHS_MISSING",
            "message": "PSF wavelengths are required for band resampling.",
            "psf_bands": psf_bands,
            "hsi_bands": hsi_bands,
        }
    return {"status": "valid", "psf_bands": psf_bands, "hsi_bands": hsi_bands, "resampled": True}


def align_psf_to_hsi_wavelengths(
    psf_cube: np.ndarray,
    psf_wavelengths: list[float],
    hsi_wavelengths: list[float],
) -> np.ndarray:
    cube = np.asarray(psf_cube, dtype=np.float32)
    if cube.ndim != 4:
        raise ValueError(f"Expected PSF cube [D,B,H,W], got {cube.shape}")
    psf_w = np.asarray(psf_wavelengths, dtype=np.float64)
    hsi_w = np.asarray(hsi_wavelengths, dtype=np.float64)
    if len(psf_w) != cube.shape[1]:
        raise ValueError("psf_wavelengths length must match PSF band count")
    D, _, H, W = cube.shape
    flat = cube.transpose(0, 2, 3, 1).reshape(-1, cube.shape[1])
    aligned = np.empty((flat.shape[0], len(hsi_w)), dtype=np.float32)
    for idx, values in enumerate(flat):
        aligned[idx] = np.interp(hsi_w, psf_w, values).astype(np.float32)
    return aligned.reshape(D, H, W, len(hsi_w)).transpose(0, 3, 1, 2).astype(np.float32)


def _read_wavelength_manifest(psf_path: Path) -> dict[str, Any]:
    for candidate in (psf_path.with_name("run_manifest.json"), psf_path.with_name("forward_model_manifest.json")):
        if candidate.exists():
            try:
                return json.loads(candidate.read_text(encoding="utf-8"))
            except Exception:
                return {}
    return {}
