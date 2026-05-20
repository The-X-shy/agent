"""Pure numpy HSI reconstruction metrics."""

from __future__ import annotations

import numpy as np


def psnr(pred: np.ndarray, target: np.ndarray, max_value: float = 1.0) -> float:
    mse = float(np.mean((np.asarray(pred) - np.asarray(target)) ** 2))
    if mse <= 1e-12:
        return 99.0
    return round(float(20.0 * np.log10(max_value) - 10.0 * np.log10(mse)), 6)


def ssim_simple(pred: np.ndarray, target: np.ndarray) -> float:
    x = np.asarray(pred, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    c1 = 0.01**2
    c2 = 0.03**2
    mux = float(x.mean())
    muy = float(y.mean())
    vx = float(x.var())
    vy = float(y.var())
    cov = float(((x - mux) * (y - muy)).mean())
    score = ((2 * mux * muy + c1) * (2 * cov + c2)) / max((mux**2 + muy**2 + c1) * (vx + vy + c2), 1e-12)
    return round(float(np.clip(score, 0.0, 1.0)), 6)


def sam(pred: np.ndarray, target: np.ndarray) -> float:
    x = np.asarray(pred, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if x.ndim == 3:
        x = x[None]
        y = y[None]
    x_vec = np.moveaxis(x, 1, -1).reshape(-1, x.shape[1])
    y_vec = np.moveaxis(y, 1, -1).reshape(-1, y.shape[1])
    dot = np.sum(x_vec * y_vec, axis=1)
    denom = np.linalg.norm(x_vec, axis=1) * np.linalg.norm(y_vec, axis=1)
    cos = np.clip(dot / np.maximum(denom, 1e-12), -1.0, 1.0)
    return round(float(np.mean(np.arccos(cos))), 6)


def ergas(pred: np.ndarray, target: np.ndarray) -> float:
    rmse = np.sqrt(np.mean((np.asarray(pred) - np.asarray(target)) ** 2, axis=(0, 2, 3)))
    means = np.mean(np.asarray(target), axis=(0, 2, 3))
    value = 100.0 * np.sqrt(np.mean((rmse / np.maximum(means, 1e-8)) ** 2))
    return round(float(value), 6)


def per_band_rmse(pred: np.ndarray, target: np.ndarray) -> list[float]:
    values = np.sqrt(np.mean((np.asarray(pred) - np.asarray(target)) ** 2, axis=(0, 2, 3)))
    return [round(float(item), 6) for item in values]


def worst_depth_sam(pred: np.ndarray, target: np.ndarray, depth_indices) -> float:
    depths = np.asarray(depth_indices)
    worst = 0.0
    for depth in sorted(set(int(item) for item in depths.tolist())):
        mask = depths == depth
        if np.any(mask):
            worst = max(worst, sam(np.asarray(pred)[mask], np.asarray(target)[mask]))
    return round(float(worst), 6)


def metric_summary(pred: np.ndarray, target: np.ndarray, depth_indices) -> dict:
    return {
        "PSNR": psnr(pred, target),
        "SSIM": ssim_simple(pred, target),
        "SAM": sam(pred, target),
        "ERGAS": ergas(pred, target),
        "per_band_RMSE": per_band_rmse(pred, target),
        "worst_depth_SAM": worst_depth_sam(pred, target, depth_indices),
    }
