"""HSI reconstruction baselines."""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass, field
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import numpy as np

from optiresearch.hsi.metrics import metric_summary


@dataclass
class ReconstructorResult:
    status: str
    metrics: dict[str, Any] = field(default_factory=dict)
    artifact_paths: list[str] = field(default_factory=list)
    error_code: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def torch_available() -> bool:
    return find_spec("torch") is not None


OPTICAL_SCALAR_FEATURES = (
    "spectral_separability_score",
    "depth_stability_score",
    "coding_strength",
    "band_condition_score",
)


def build_optical_feature_maps(measurements: np.ndarray, optical_features: dict | None = None) -> np.ndarray:
    measurements = np.asarray(measurements)
    if measurements.ndim != 4:
        raise ValueError(f"Expected measurements [N, C, H, W], got {measurements.shape}")
    feats = optical_features or {}
    N, _, H, W = measurements.shape
    maps = np.zeros((N, len(OPTICAL_SCALAR_FEATURES), H, W), dtype=np.float64)
    for idx, key in enumerate(OPTICAL_SCALAR_FEATURES):
        maps[:, idx, :, :] = float(feats.get(key, 0.0))
    return maps


def _maybe_concat_optical_maps(measurements: np.ndarray, optical_features: dict | None, enabled: bool, injection: str) -> np.ndarray:
    arr = np.asarray(measurements, dtype=np.float32)
    if not enabled or injection != "concat_scalar_maps":
        return arr
    maps = build_optical_feature_maps(arr, optical_features).astype(np.float32)
    return np.concatenate([arr, maps], axis=1).astype(np.float32)


class LinearSpectralReconstructor:
    def __init__(self, output_bands: int) -> None:
        self.output_bands = output_bands
        self.weights = np.ones(output_bands, dtype=np.float32)

    def fit(self, train_measurements: np.ndarray, train_targets: np.ndarray) -> "LinearSpectralReconstructor":
        x = np.asarray(train_measurements, dtype=np.float32)[:, 0]
        y = np.asarray(train_targets, dtype=np.float32)
        denom = float(np.sum(x * x)) + 1e-8
        weights = []
        for band in range(y.shape[1]):
            weights.append(float(np.sum(x * y[:, band]) / denom))
        self.weights = np.asarray(weights, dtype=np.float32)
        self.output_bands = int(y.shape[1])
        return self

    def predict(self, measurements: np.ndarray) -> np.ndarray:
        x = np.asarray(measurements, dtype=np.float32)[:, 0]
        return (x[:, None, :, :] * self.weights[None, :, None, None]).astype(np.float32)


class OpticalConditionedLinearReconstructor:
    """Band-conditioned linear reconstructor that uses optical features to create
    per-band spatial filter banks, making reconstruction sensitive to encoder PSF differences."""

    def __init__(self, output_bands: int, regularization: float = 1e-3) -> None:
        self.output_bands = output_bands
        self.regularization = regularization
        self.weights = np.ones(output_bands, dtype=np.float32)
        self.per_band_weights = None

    def fit(
        self,
        train_measurements: np.ndarray,
        train_targets: np.ndarray,
        optical_features: dict | None = None,
    ) -> "OpticalConditionedLinearReconstructor":
        x = np.asarray(train_measurements, dtype=np.float32)[:, 0]
        y = np.asarray(train_targets, dtype=np.float32)
        N, H, W = x.shape
        B = y.shape[1]
        self.output_bands = B

        feats = optical_features or {}
        spectral_sep = float(feats.get("spectral_separability_score", 0.3))
        depth_stability = float(feats.get("depth_stability_score", 0.8))
        spread = np.asarray(feats.get("band_spread", np.ones(B) * 0.2), dtype=np.float64)
        centroids_x = np.asarray(feats.get("band_centroid_x", np.zeros(B)), dtype=np.float64)
        centroids_y = np.asarray(feats.get("band_centroid_y", np.zeros(B)), dtype=np.float64)

        spread_mean = max(float(spread.mean()), 1e-8)

        self.per_band_weights = []
        for band in range(B):
            y_band_flat = y[:, band].reshape(N, -1)
            x_flat = x.reshape(N, -1)

            effective_sep = 0.1 + 0.9 * spectral_sep

            if effective_sep < 0.3:
                design = np.ones((N, 1), dtype=np.float64)
            elif effective_sep < 0.5:
                design = np.column_stack([np.ones(N, dtype=np.float64), x_flat.mean(axis=1)])
            else:
                blur_sigma = 0.5 + effective_sep * 2.0
                blurred = np.array([_gaussian_blur_2d(x[i], blur_sigma) for i in range(N)])
                design = np.column_stack([
                    np.ones(N, dtype=np.float64),
                    x_flat.mean(axis=1),
                    blurred.reshape(N, -1).mean(axis=1),
                    blurred.reshape(N, -1).std(axis=1),
                ])

            reg = self.regularization * (1.0 + (1.0 - effective_sep))
            gram = design.T @ design + reg * np.eye(design.shape[1])
            coeffs = np.linalg.solve(gram, design.T @ y_band_flat)
            self.per_band_weights.append({"coeffs": coeffs.astype(np.float32), "design_cols": design.shape[1]})

        return self

    def predict(self, measurements: np.ndarray, optical_features: dict | None = None) -> np.ndarray:
        x = np.asarray(measurements, dtype=np.float32)[:, 0]
        N, H, W = x.shape
        B = self.output_bands

        feats = optical_features or {}
        spectral_sep = float(feats.get("spectral_separability_score", 0.3))

        prediction = np.zeros((N, B, H, W), dtype=np.float64)
        for band in range(B):
            entry = self.per_band_weights[band] if self.per_band_weights else {"coeffs": np.ones(1, dtype=np.float32), "design_cols": 1}
            coeffs = entry["coeffs"].astype(np.float64)
            n_cols = entry["design_cols"]
            x_flat = x.reshape(N, -1)

            effective_sep = 0.1 + 0.9 * spectral_sep
            if n_cols == 1:
                design = np.ones((N, 1), dtype=np.float64)
            elif n_cols == 2:
                design = np.column_stack([np.ones(N, dtype=np.float64), x_flat.mean(axis=1)])
            else:
                blur_sigma = 0.5 + effective_sep * 2.0
                blurred = np.array([_gaussian_blur_2d(x[i], blur_sigma) for i in range(N)])
                design = np.column_stack([
                    np.ones(N, dtype=np.float64),
                    x_flat.mean(axis=1),
                    blurred.reshape(N, -1).mean(axis=1),
                    blurred.reshape(N, -1).std(axis=1),
                ])

            band_pred_flat = design @ coeffs
            prediction[:, band] = band_pred_flat.reshape(N, H, W)
        return np.clip(prediction, 0.0, 1.0).astype(np.float32)


def _gaussian_blur_2d(image: np.ndarray, sigma: float) -> np.ndarray:
    if sigma < 0.2:
        return image
    ksize = max(3, int(np.ceil(sigma * 3)) * 2 + 1)
    ax = np.arange(ksize, dtype=np.float64) - ksize // 2
    kernel = np.exp(-(ax[:, None]**2 + ax[None, :]**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
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


def _shift_image_2d(image: np.ndarray, sx: float, sy: float) -> np.ndarray:
    if abs(sx) < 0.5 and abs(sy) < 0.5:
        return image
    ix = int(round(sx))
    iy = int(round(sy))
    if ix == 0 and iy == 0:
        return image
    result = np.zeros_like(image)
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


class TinyCNNReconstructor:
    """Minimal CNN reconstructor. Requires PyTorch. Falls back gracefully when unavailable."""

    def __init__(
        self,
        output_bands: int,
        hidden_channels: int = 32,
        depth: int = 4,
        epochs: int = 5,
        lr: float = 1e-3,
        batch_size: int = 4,
        seed: int = 42,
        device: str = "cpu",
        input_channels: int = 1,
        spectral_smoothness_weight: float = 0.0,
    ) -> None:
        self.output_bands = output_bands
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.depth = depth
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.seed = seed
        self.device = device
        self.spectral_smoothness_weight = spectral_smoothness_weight
        self.model = None
        self.error = None
        self._trained = False
        self._has_torch = torch_available()
        if not self._has_torch:
            self._has_torch = False
            self.error = {"code": "TORCH_NOT_AVAILABLE", "message": "PyTorch is not installed."}

    def available(self) -> bool:
        return self._has_torch

    def fit(
        self,
        train_measurements: np.ndarray,
        train_targets: np.ndarray,
        optical_features: dict | None = None,
        epochs: int | None = None,
        learning_rate: float | None = None,
        verbose: bool = False,
    ) -> "TinyCNNReconstructor":
        if not self._has_torch:
            return self
        x = np.asarray(train_measurements, dtype=np.float32)
        self.input_channels = int(x.shape[1])
        self._train_model(
            x,
            np.asarray(train_targets, dtype=np.float32),
            epochs=epochs or self.epochs,
            learning_rate=learning_rate or self.lr,
        )
        self._trained = True
        if verbose:
            print(f"[TinyCNN] trained {epochs or self.epochs} epochs")
        return self

    def predict(self, measurements: np.ndarray, optical_features: dict | None = None) -> np.ndarray:
        if not self._has_torch or self.model is None:
            if self.error:
                raise RuntimeError(f"TinyCNN not available: {self.error['code']}")
            raise RuntimeError("TinyCNN model not trained")
        import torch
        device = next(self.model.parameters()).device
        self.model.eval()
        with torch.no_grad():
            x = torch.from_numpy(np.asarray(measurements, dtype=np.float32)).to(device)
            pred = self.model(x).cpu().numpy()
        return np.clip(pred, 0.0, 1.0).astype(np.float32)

    def run(
        self,
        train_measurements: np.ndarray,
        train_targets: np.ndarray,
        test_measurements: np.ndarray,
        test_targets: np.ndarray,
        test_depth_indices,
        output_dir: Path,
        optical_features: dict | None = None,
        optical_feature_injection: str = "none",
    ) -> ReconstructorResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        if not self.available():
            manifest = self._manifest("skipped", optical_feature_injection, error_code="TORCH_NOT_AVAILABLE")
            manifest_path = output_dir / "reconstruction_manifest.json"
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
            return ReconstructorResult(
                status="skipped",
                metrics={"network_type": "tiny_cnn", "output_bands": int(self.output_bands)},
                artifact_paths=[str(manifest_path)],
                error_code="TORCH_NOT_AVAILABLE",
                metadata=manifest,
            )

        training_log = self._train_model(
            np.asarray(train_measurements, dtype=np.float32),
            np.asarray(train_targets, dtype=np.float32),
            epochs=self.epochs,
            learning_rate=self.lr,
        )
        self._trained = True
        prediction = self.predict(test_measurements)
        metrics = metric_summary(prediction, test_targets, test_depth_indices)
        metrics.update({"network_type": "tiny_cnn", "output_bands": int(test_targets.shape[1])})
        paths = _save_reconstruction_outputs(output_dir, prediction, test_targets, metrics)
        import torch

        checkpoint_path = output_dir / "checkpoint.pt"
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "config": self._config(),
            },
            checkpoint_path,
        )
        training_log_path = output_dir / "training_log.json"
        training_log_path.write_text(json.dumps(training_log, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
        manifest = self._manifest("succeeded", optical_feature_injection, metrics=metrics)
        manifest_path = output_dir / "reconstruction_manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
        artifact_paths = [*paths, checkpoint_path, training_log_path, manifest_path]
        return ReconstructorResult(
            status="succeeded",
            metrics=metrics,
            artifact_paths=[str(path) for path in artifact_paths],
            metadata=manifest,
        )

    def _build_model(self):
        import torch.nn as nn

        layers: list[Any] = []
        in_ch = self.input_channels
        for idx in range(max(1, self.depth - 1)):
            out_ch = self.hidden_channels
            layers.append(nn.Conv2d(in_ch, out_ch, 3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            in_ch = out_ch
        layers.append(nn.Conv2d(in_ch, self.output_bands, 3, padding=1))
        return nn.Sequential(*layers).to(self._torch_device())

    def _train_model(self, x_np: np.ndarray, y_np: np.ndarray, epochs: int, learning_rate: float) -> list[dict[str, float]]:
        import torch
        import torch.nn as nn

        _seed_torch(self.seed)
        if self.model is None:
            self.input_channels = int(x_np.shape[1])
            self.model = self._build_model()
        device = self._torch_device()
        x = torch.from_numpy(x_np).to(device)
        y = torch.from_numpy(y_np).to(device)
        opt = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()
        log: list[dict[str, float]] = []
        self.model.train()
        indices = np.arange(x.shape[0])
        for epoch in range(int(epochs)):
            rng = np.random.default_rng(self.seed + epoch)
            rng.shuffle(indices)
            losses = []
            for start in range(0, len(indices), max(1, self.batch_size)):
                batch_idx = indices[start:start + max(1, self.batch_size)]
                xb = x[batch_idx]
                yb = y[batch_idx]
                opt.zero_grad()
                pred = self.model(xb)
                loss = loss_fn(pred, yb)
                if self.spectral_smoothness_weight > 0 and pred.shape[1] > 1:
                    smooth = torch.mean((pred[:, 1:] - pred[:, :-1]) ** 2)
                    loss = loss + self.spectral_smoothness_weight * smooth
                loss.backward()
                opt.step()
                losses.append(float(loss.detach().cpu().item()))
            log.append({"epoch": float(epoch + 1), "loss": float(np.mean(losses)) if losses else 0.0})
        return log

    def _torch_device(self):
        import torch

        if self.device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _config(self) -> dict[str, Any]:
        return {
            "network_type": "tiny_cnn",
            "output_bands": int(self.output_bands),
            "input_channels": int(self.input_channels),
            "hidden_channels": int(self.hidden_channels),
            "depth": int(self.depth),
            "epochs": int(self.epochs),
            "lr": float(self.lr),
            "batch_size": int(self.batch_size),
            "seed": int(self.seed),
            "device": self.device,
        }

    def _manifest(self, status: str, optical_feature_injection: str, error_code: str | None = None, metrics: dict | None = None) -> dict[str, Any]:
        return {
            **self._config(),
            "status": status,
            "error_code": error_code,
            "metrics": metrics or {},
            "optical_feature_injection": optical_feature_injection,
        }


class UNetTinyReconstructor(TinyCNNReconstructor):
    """Small UNet-style optional PyTorch reconstructor with the same result contract."""

    def _build_model(self):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        hidden = self.hidden_channels

        class _UNetTiny(nn.Module):
            def __init__(self, in_ch: int, out_ch: int):
                super().__init__()
                self.enc1 = nn.Sequential(nn.Conv2d(in_ch, hidden, 3, padding=1), nn.ReLU(inplace=True))
                self.enc2 = nn.Sequential(nn.Conv2d(hidden, hidden * 2, 3, padding=1), nn.ReLU(inplace=True))
                self.mid = nn.Sequential(nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1), nn.ReLU(inplace=True))
                self.dec = nn.Sequential(nn.Conv2d(hidden * 3, hidden, 3, padding=1), nn.ReLU(inplace=True))
                self.out = nn.Conv2d(hidden, out_ch, 3, padding=1)

            def forward(self, x):
                e1 = self.enc1(x)
                e2 = self.enc2(F.avg_pool2d(e1, 2))
                m = self.mid(e2)
                up = F.interpolate(m, size=e1.shape[-2:], mode="nearest")
                return self.out(self.dec(torch.cat([up, e1], dim=1)))

        return _UNetTiny(self.input_channels, self.output_bands).to(self._torch_device())

    def run(
        self,
        train_measurements: np.ndarray,
        train_targets: np.ndarray,
        test_measurements: np.ndarray,
        test_targets: np.ndarray,
        test_depth_indices,
        output_dir: Path,
        optical_features: dict | None = None,
        optical_feature_injection: str = "none",
    ) -> ReconstructorResult:
        result = super().run(
            train_measurements,
            train_targets,
            test_measurements,
            test_targets,
            test_depth_indices,
            output_dir,
            optical_features,
            optical_feature_injection,
        )
        result.metrics["network_type"] = "unet_tiny"
        result.metadata["network_type"] = "unet_tiny"
        manifest_path = output_dir / "reconstruction_manifest.json"
        if manifest_path.exists():
            manifest_path.write_text(json.dumps(result.metadata, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
        return result

    def _config(self) -> dict[str, Any]:
        config = super()._config()
        config["network_type"] = "unet_tiny"
        return config


def _seed_torch(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if not torch_available():
        return
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_linear_reconstruction(
    train_measurements: np.ndarray,
    train_targets: np.ndarray,
    test_measurements: np.ndarray,
    test_targets: np.ndarray,
    test_depth_indices,
    output_dir: Path,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    reconstructor = LinearSpectralReconstructor(output_bands=train_targets.shape[1]).fit(train_measurements, train_targets)
    prediction = reconstructor.predict(test_measurements)
    metrics = metric_summary(prediction, test_targets, test_depth_indices)
    metrics.update({"network_type": "linear_baseline", "output_bands": int(test_targets.shape[1])})
    np.savez_compressed(output_dir / "reconstructed_test.npz", reconstruction=prediction, target=test_targets)
    (output_dir / "reconstruction_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    with (output_dir / "spectral_curves.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["band", "weight"])
        for idx, value in enumerate(reconstructor.weights.tolist()):
            writer.writerow([idx, round(float(value), 8)])
    manifest = {"network_type": "linear_baseline", "weights": [round(float(item), 8) for item in reconstructor.weights.tolist()]}
    (output_dir / "reconstruction_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    return {
        "metrics": metrics,
        "artifacts": [
            output_dir / "reconstruction_metrics.json",
            output_dir / "reconstructed_test.npz",
            output_dir / "spectral_curves.csv",
            output_dir / "reconstruction_manifest.json",
        ],
        "prediction": prediction,
    }


def run_reconstruction(
    reconstructor_type: str,
    train_measurements: np.ndarray,
    train_targets: np.ndarray,
    test_measurements: np.ndarray,
    test_targets: np.ndarray,
    test_depth_indices,
    output_dir: Path,
    optical_features: dict | None = None,
    use_optical_feature_maps: bool = False,
    optical_feature_injection: str = "none",
    train_options: dict[str, Any] | None = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_options = train_options or {}
    train_x = _maybe_concat_optical_maps(train_measurements, optical_features, use_optical_feature_maps, optical_feature_injection)
    test_x = _maybe_concat_optical_maps(test_measurements, optical_features, use_optical_feature_maps, optical_feature_injection)
    input_channels = int(train_x.shape[1])

    if reconstructor_type == "linear_baseline":
        reconstructor = LinearSpectralReconstructor(output_bands=train_targets.shape[1])
        reconstructor.fit(train_measurements, train_targets)
        prediction = reconstructor.predict(test_measurements)
        weights_list = [round(float(w), 8) for w in reconstructor.weights.tolist()]
    elif reconstructor_type == "optical_conditioned_linear":
        reconstructor = OpticalConditionedLinearReconstructor(output_bands=train_targets.shape[1])
        reconstructor.fit(train_measurements, train_targets, optical_features)
        prediction = reconstructor.predict(test_measurements, optical_features)
        weights_list = []
    elif reconstructor_type == "tiny_cnn":
        reconstructor = TinyCNNReconstructor(
            output_bands=train_targets.shape[1],
            input_channels=input_channels,
            hidden_channels=int(train_options.get("hidden_channels", train_options.get("tiny_cnn_hidden", 32))),
            depth=int(train_options.get("depth", 4)),
            epochs=int(train_options.get("epochs", train_options.get("tiny_cnn_epochs", 5))),
            lr=float(train_options.get("lr", 1e-3)),
            batch_size=int(train_options.get("batch_size", 4)),
            seed=int(train_options.get("seed", 42)),
            device=str(train_options.get("device", "cpu")),
            spectral_smoothness_weight=float(train_options.get("spectral_smoothness_weight", 0.0)),
        )
        result = reconstructor.run(train_x, train_targets, test_x, test_targets, test_depth_indices, output_dir, optical_features, optical_feature_injection)
        return _result_to_dict(result)
    elif reconstructor_type == "unet_tiny":
        reconstructor = UNetTinyReconstructor(
            output_bands=train_targets.shape[1],
            input_channels=input_channels,
            hidden_channels=int(train_options.get("hidden_channels", train_options.get("tiny_cnn_hidden", 16))),
            depth=int(train_options.get("depth", 3)),
            epochs=int(train_options.get("epochs", train_options.get("tiny_cnn_epochs", 5))),
            lr=float(train_options.get("lr", 1e-3)),
            batch_size=int(train_options.get("batch_size", 4)),
            seed=int(train_options.get("seed", 42)),
            device=str(train_options.get("device", "cpu")),
        )
        result = reconstructor.run(train_x, train_targets, test_x, test_targets, test_depth_indices, output_dir, optical_features, optical_feature_injection)
        return _result_to_dict(result)
    else:
        raise ValueError(f"Unknown reconstructor_type: {reconstructor_type}")

    metrics = metric_summary(prediction, test_targets, test_depth_indices)
    metrics.update({"network_type": reconstructor_type, "output_bands": int(test_targets.shape[1])})
    paths = _save_reconstruction_outputs(output_dir, prediction, test_targets, metrics)
    if weights_list:
        with (output_dir / "spectral_curves.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["band", "weight"])
            for idx, value in enumerate(weights_list):
                writer.writerow([idx, value])
    manifest = {
        "network_type": reconstructor_type,
        "status": "succeeded",
        "weights": weights_list if weights_list else None,
        "input_channels": input_channels,
        "output_bands": int(test_targets.shape[1]),
        "use_optical_features": bool(use_optical_feature_maps),
        "optical_feature_injection": optical_feature_injection if use_optical_feature_maps else "none",
    }
    manifest_path = output_dir / "reconstruction_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    artifacts = [*paths, manifest_path]
    if weights_list:
        artifacts.append(output_dir / "spectral_curves.csv")
    return {
        "status": "succeeded",
        "metrics": metrics,
        "artifacts": artifacts,
        "prediction": prediction,
        "error_code": None,
        "metadata": manifest,
    }


def _save_reconstruction_outputs(output_dir: Path, prediction: np.ndarray, target: np.ndarray, metrics: dict[str, Any]) -> list[Path]:
    np.savez_compressed(output_dir / "reconstructed_test.npz", reconstruction=prediction, target=target)
    (output_dir / "reconstruction_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    return [output_dir / "reconstruction_metrics.json", output_dir / "reconstructed_test.npz"]


def _result_to_dict(result: ReconstructorResult) -> dict[str, Any]:
    payload = {
        "status": result.status,
        "metrics": result.metrics,
        "artifacts": [Path(path) for path in result.artifact_paths],
        "prediction": None,
        "error_code": result.error_code,
        "metadata": result.metadata,
    }
    reconstructed = next((Path(path) for path in result.artifact_paths if Path(path).name == "reconstructed_test.npz"), None)
    if reconstructed and reconstructed.exists():
        payload["prediction"] = np.load(reconstructed)["reconstruction"]
    return payload
