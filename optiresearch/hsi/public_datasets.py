"""Local-path HSI dataset adapters for synthetic and public-placeholder datasets."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from optiresearch.hsi.dataset import SyntheticHSIDataset
from optiresearch.schemas.hsi import HSIDatasetSpec, build_default_synthetic_hsi_dataset_spec, validate_hsi_dataset_spec


class HSIDatasetAdapter:
    dataset_id = "base"

    def available(self) -> bool:
        raise NotImplementedError

    def prepare(self, output_dir: Path) -> dict:
        raise NotImplementedError

    def load_split(self, split: str) -> dict:
        raise NotImplementedError


class SyntheticAdapter(HSIDatasetAdapter):
    dataset_id = "synthetic"

    def __init__(self, spec: HSIDatasetSpec | None = None, seed: int = 42) -> None:
        self.spec = spec or build_default_synthetic_hsi_dataset_spec()
        self.seed = seed
        self.prepared_dir: Path | None = None

    def available(self) -> bool:
        return True

    def prepare(self, output_dir: Path) -> dict:
        output_dir = Path(output_dir)
        dataset = SyntheticHSIDataset(self.spec, seed=self.seed)
        manifest = dataset.save(output_dir)
        self.prepared_dir = output_dir
        payload = dict(manifest)
        payload.update(
            {
                "status": "prepared",
                "dataset_id": self.dataset_id,
                "dataset_family": "synthetic",
                "output_dir": str(output_dir),
                "validation": validate_hsi_dataset_spec(manifest),
            }
        )
        (output_dir / "dataset_manifest.json").write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
        return payload

    def load_split(self, split: str) -> dict:
        if self.prepared_dir is None:
            raise RuntimeError("SyntheticAdapter.prepare must be called before load_split")
        return _load_npz_split(self.prepared_dir / f"{split}.npz")


class LocalNPZHSIAdapter(HSIDatasetAdapter):
    dataset_id = "local_npz"
    env_var = "OPTIRESEARCH_HSI_DATASET_PATH"

    def __init__(
        self,
        path: str | Path | None = None,
        dataset_id: str | None = None,
        dataset_family: str = "local_npz",
        crop_size: int = 32,
        patch_stride: int = 32,
        normalization: str = "per_band",
    ) -> None:
        env_path = os.getenv(self.env_var)
        self.path = Path(path or env_path).expanduser() if path or env_path else None
        self.dataset_id = dataset_id or self.dataset_id
        self.dataset_family = dataset_family
        self.crop_size = int(crop_size)
        self.patch_stride = int(patch_stride)
        self.normalization = normalization
        self.prepared_dir: Path | None = None

    def available(self) -> bool:
        if self.path is None or not self.path.exists():
            return False
        return self._detect_format()["status"] == "ok"

    def prepare(self, output_dir: Path) -> dict:
        output_dir = Path(output_dir)
        if self.path is None:
            return self._error("DATASET_NOT_CONFIGURED", "Set OPTIRESEARCH_HSI_DATASET_PATH or pass --path.", output_dir)
        if not self.path.exists():
            return self._error("DATASET_PATH_NOT_FOUND", f"Dataset path does not exist: {self.path}", output_dir)

        try:
            detected = self._detect_format()
            if detected["status"] != "ok":
                return self._error(detected["error_code"], detected["message"], output_dir)
            arrays = self._load_detected_format(detected)
            validation = validate_dataset_shapes({split: payload["hsi"] for split, payload in arrays.items()})
            if validation["status"] != "valid":
                return self._error(validation["error_code"], validation["message"], output_dir)
        except Exception as exc:
            return self._error("DATASET_READ_FAILED", str(exc), output_dir)

        manifest = save_prepared_dataset(
            arrays,
            output_dir,
            dataset_id=self.dataset_id,
            dataset_family=self.dataset_family,
            source_path=self.path,
            normalization=self.normalization,
            crop_size=self.crop_size,
            patch_stride=self.patch_stride,
            license_note="Local-path dataset. User is responsible for dataset license and provenance.",
            metadata={"input_format": detected["format"], "public_download_performed": False},
        )
        self.prepared_dir = output_dir
        return manifest

    def load_split(self, split: str) -> dict:
        root = self.prepared_dir or self.path
        if root is None:
            raise RuntimeError("LocalNPZHSIAdapter has no configured path")
        return _load_npz_split(root / f"{split}.npz")

    def _error(self, code: str, message: str, output_dir: Path | None = None) -> dict:
        return {
            "status": "error",
            "dataset_id": self.dataset_id,
            "dataset_family": self.dataset_family,
            "error_code": code,
            "message": message,
            "path": str(self.path) if self.path else None,
            "output_dir": str(output_dir) if output_dir else None,
        }

    def _detect_format(self) -> dict[str, Any]:
        path = self.path
        if path is None:
            return {"status": "error", "error_code": "DATASET_NOT_CONFIGURED", "message": "No path configured."}
        if path.is_file():
            if path.suffix.lower() in {".npz", ".npy"}:
                return {"status": "ok", "format": "single_cube", "path": path}
            return {"status": "error", "error_code": "UNSUPPORTED_DATASET_FILE", "message": f"Unsupported file: {path}"}
        if all((path / f"{split}.npz").exists() for split in _SPLITS):
            return {"status": "ok", "format": "split_npz", "path": path}
        if (path / "dataset.npz").exists():
            return {"status": "ok", "format": "dataset_npz", "path": path / "dataset.npz"}
        for candidate in sorted(path.glob("*.npz")) + sorted(path.glob("*.npy")):
            return {"status": "ok", "format": "single_cube", "path": candidate}
        return {
            "status": "error",
            "error_code": "DATASET_SPLITS_MISSING",
            "message": "Expected train/val/test .npz, dataset.npz, or a single cube .npz/.npy file.",
        }

    def _load_detected_format(self, detected: dict[str, Any]) -> dict[str, dict[str, Any]]:
        fmt = detected["format"]
        if fmt == "split_npz":
            return self._load_format_a(Path(detected["path"]))
        if fmt == "dataset_npz":
            return self._load_format_b(Path(detected["path"]))
        if fmt == "single_cube":
            return self._load_format_c(Path(detected["path"]))
        raise ValueError(f"Unsupported local NPZ format: {fmt}")

    def _load_format_a(self, root: Path) -> dict[str, dict[str, Any]]:
        arrays: dict[str, dict[str, Any]] = {}
        for split in _SPLITS:
            payload = _load_npz_split(root / f"{split}.npz")
            if "hsi" not in payload:
                raise ValueError(f"{split}.npz must contain hsi.")
            arrays[split] = _prepare_hsi_payload(
                payload["hsi"],
                crop_size=self.crop_size,
                stride=self.patch_stride,
                normalization=self.normalization,
                depth_indices=payload.get("depth_indices"),
                wavelengths_nm=_payload_wavelengths(payload),
                masks=payload.get("masks"),
            )
        return arrays

    def _load_format_b(self, path: Path) -> dict[str, dict[str, Any]]:
        payload = _load_npz_split(path)
        if "hsi" not in payload:
            raise ValueError("dataset.npz must contain hsi.")
        hsi = _ensure_nbhw(payload["hsi"])
        split_values = payload.get("split")
        if split_values is None:
            split_values = _default_split_labels(hsi.shape[0])
        split_values = np.asarray(split_values).astype(str)
        arrays = {}
        for split in _SPLITS:
            mask = split_values == split
            arrays[split] = _prepare_hsi_payload(
                hsi[mask],
                crop_size=self.crop_size,
                stride=self.patch_stride,
                normalization=self.normalization,
                depth_indices=payload.get("depth_indices")[mask] if payload.get("depth_indices") is not None else None,
                wavelengths_nm=_payload_wavelengths(payload),
                masks=payload.get("masks")[mask] if payload.get("masks") is not None and payload.get("masks").shape[0] == hsi.shape[0] else None,
            )
        return _ensure_nonempty_splits(arrays)

    def _load_format_c(self, path: Path) -> dict[str, dict[str, Any]]:
        if path.suffix.lower() == ".npy":
            payload = {"cube": np.load(path)}
        else:
            payload = _load_npz_split(path)
        cube = payload.get("cube")
        if cube is None:
            cube = payload.get("hsi")
        if cube is None:
            first_key = next(iter(payload))
            cube = payload[first_key]
        patches = _prepare_hsi_payload(
            cube,
            crop_size=self.crop_size,
            stride=self.patch_stride,
            normalization=self.normalization,
            wavelengths_nm=_payload_wavelengths(payload),
        )
        labels = _default_split_labels(patches["hsi"].shape[0])
        arrays = {}
        for split in _SPLITS:
            mask = labels == split
            arrays[split] = {
                "hsi": patches["hsi"][mask],
                "depth_indices": patches["depth_indices"][mask],
            }
            if "wavelengths_nm" in patches:
                arrays[split]["wavelengths_nm"] = patches["wavelengths_nm"]
        return _ensure_nonempty_splits(arrays)


class CAVEPlaceholderAdapter(LocalNPZHSIAdapter):
    dataset_id = "cave"
    env_var = "OPTIRESEARCH_CAVE_PATH"

    def __init__(self, path: str | Path | None = None, crop_size: int = 32, patch_stride: int = 32, normalization: str = "per_band") -> None:
        env_path = os.getenv(self.env_var)
        super().__init__(path or env_path, dataset_id="cave", dataset_family="cave", crop_size=crop_size, patch_stride=patch_stride, normalization=normalization)

    def prepare(self, output_dir: Path) -> dict:
        if self.path is None:
            return self._not_configured(output_dir)
        if not self.path.exists():
            return self._error("DATASET_PATH_NOT_FOUND", f"CAVE path does not exist: {self.path}", Path(output_dir))
        return super().prepare(output_dir)

    def _not_configured(self, output_dir: Path) -> dict:
        return {
            "status": "error",
            "dataset_id": self.dataset_id,
            "dataset_family": "cave",
            "error_code": "DATASET_NOT_CONFIGURED",
            "message": "Set OPTIRESEARCH_CAVE_PATH to a local directory containing train.npz, val.npz, and test.npz.",
            "expected_structure": expected_npz_structure("cave"),
            "import_instructions": "Convert the CAVE HSI scenes into local train/val/test .npz files with hsi shaped [N, B, H, W]. No automatic download is performed.",
            "output_dir": str(output_dir),
        }


class ICVLPlaceholderAdapter(LocalNPZHSIAdapter):
    dataset_id = "icvl"
    env_var = "OPTIRESEARCH_ICVL_PATH"

    def __init__(self, path: str | Path | None = None, crop_size: int = 32, patch_stride: int = 32, normalization: str = "per_band") -> None:
        env_path = os.getenv(self.env_var)
        super().__init__(path or env_path, dataset_id="icvl", dataset_family="icvl", crop_size=crop_size, patch_stride=patch_stride, normalization=normalization)

    def prepare(self, output_dir: Path) -> dict:
        if self.path is None:
            return self._not_configured(output_dir)
        if not self.path.exists():
            return self._error("DATASET_PATH_NOT_FOUND", f"ICVL path does not exist: {self.path}", Path(output_dir))
        return super().prepare(output_dir)

    def _not_configured(self, output_dir: Path) -> dict:
        return {
            "status": "error",
            "dataset_id": self.dataset_id,
            "dataset_family": "icvl",
            "error_code": "DATASET_NOT_CONFIGURED",
            "message": "Set OPTIRESEARCH_ICVL_PATH to a local directory containing train.npz, val.npz, and test.npz.",
            "expected_structure": expected_npz_structure("icvl"),
            "import_instructions": "Convert the ICVL HSI scenes into local train/val/test .npz files with hsi shaped [N, B, H, W]. No automatic download is performed.",
            "output_dir": str(output_dir),
        }


def expected_npz_structure(dataset_id: str) -> dict[str, Any]:
    return {
        "dataset_id": dataset_id,
        "required_files": ["train.npz", "val.npz", "test.npz"],
        "required_arrays": {"hsi": "[N, B, H, W] float array"},
        "optional_arrays": {
            "depth_indices": "[N] int array",
            "wavelengths_nm": "[B] float array",
            "masks": "optional masks aligned with hsi samples",
        },
    }


def list_hsi_dataset_adapters() -> dict[str, dict[str, Any]]:
    try:
        from optiresearch.hsi.cave_icvl import CAVEAdapter, ICVLAdapter
    except Exception:
        CAVEAdapter = CAVEPlaceholderAdapter  # type: ignore[assignment]
        ICVLAdapter = ICVLPlaceholderAdapter  # type: ignore[assignment]
    adapters = {
        "synthetic": SyntheticAdapter(),
        "local_npz": LocalNPZHSIAdapter(),
        "cave": CAVEAdapter(),
        "icvl": ICVLAdapter(),
    }
    return {
        key: {
            "dataset_id": key,
            "available": adapter.available(),
            "expected_structure": expected_npz_structure(key),
            "download_policy": "no automatic download",
        }
        for key, adapter in adapters.items()
    }


def get_hsi_dataset_adapter(
    dataset: str,
    path: str | Path | None = None,
    spec: HSIDatasetSpec | None = None,
    crop_size: int = 32,
    patch_stride: int = 32,
    normalization: str = "per_band",
) -> HSIDatasetAdapter:
    if dataset == "synthetic":
        return SyntheticAdapter(spec=spec)
    if dataset == "local_npz":
        return LocalNPZHSIAdapter(path, crop_size=crop_size, patch_stride=patch_stride, normalization=normalization)
    if dataset == "cave":
        try:
            from optiresearch.hsi.cave_icvl import CAVEAdapter

            return CAVEAdapter(path, crop_size=crop_size, patch_stride=patch_stride, normalization=normalization)
        except Exception:
            return CAVEPlaceholderAdapter(path, crop_size=crop_size, patch_stride=patch_stride, normalization=normalization)
    if dataset == "icvl":
        try:
            from optiresearch.hsi.cave_icvl import ICVLAdapter

            return ICVLAdapter(path, crop_size=crop_size, patch_stride=patch_stride, normalization=normalization)
        except Exception:
            return ICVLPlaceholderAdapter(path, crop_size=crop_size, patch_stride=patch_stride, normalization=normalization)
    raise ValueError(f"Unknown HSI dataset adapter: {dataset}")


def _load_npz_split(path: Path) -> dict[str, Any]:
    payload = np.load(path)
    return {key: np.asarray(payload[key]) for key in payload.files}


def _read_wavelengths(path: Path) -> list[float] | None:
    payload = np.load(path)
    if "wavelengths_nm" not in payload.files:
        return None
    return [float(item) for item in np.asarray(payload["wavelengths_nm"]).reshape(-1).tolist()]


_SPLITS = ("train", "val", "test")


def infer_band_axis(cube: np.ndarray) -> int:
    arr = np.asarray(cube)
    if arr.ndim != 3:
        raise ValueError(f"Expected single HSI cube with 3 dimensions, got {arr.shape}")
    candidates = [0, 2]
    return min(candidates, key=lambda axis: arr.shape[axis])


def normalize_hsi_cube(cube: np.ndarray, mode: str = "per_band") -> np.ndarray:
    arr = np.asarray(cube, dtype=np.float32)
    if mode == "none":
        return arr.astype(np.float32)
    if mode in {"global", "minmax"}:
        mn = float(np.min(arr))
        mx = float(np.max(arr))
        return ((arr - mn) / max(mx - mn, 1e-8)).astype(np.float32)
    if mode != "per_band":
        raise ValueError(f"Unsupported HSI normalization mode: {mode}")
    if arr.ndim == 3:
        axes = (1, 2)
        mn = np.min(arr, axis=axes, keepdims=True)
        mx = np.max(arr, axis=axes, keepdims=True)
    elif arr.ndim == 4:
        axes = (0, 2, 3)
        mn = np.min(arr, axis=axes, keepdims=True)
        mx = np.max(arr, axis=axes, keepdims=True)
    else:
        raise ValueError(f"Expected 3D or 4D HSI data, got {arr.shape}")
    return ((arr - mn) / np.maximum(mx - mn, 1e-8)).astype(np.float32)


def crop_to_patches(cube: np.ndarray, crop_size: int, stride: int) -> np.ndarray:
    bhw = _ensure_bhw(cube)
    B, H, W = bhw.shape
    crop = min(int(crop_size), H, W)
    stride = max(int(stride), 1)
    patches = []
    for y in _starts(H, crop, stride):
        for x in _starts(W, crop, stride):
            patches.append(bhw[:, y:y + crop, x:x + crop])
    if not patches:
        patches.append(bhw[:, :crop, :crop])
    return np.asarray(patches, dtype=np.float32)


def validate_dataset_shapes(splits: dict[str, np.ndarray]) -> dict[str, Any]:
    missing = [split for split in _SPLITS if split not in splits]
    if missing:
        return {"status": "error", "error_code": "DATASET_SPLITS_MISSING", "message": f"Missing splits: {', '.join(missing)}"}
    shapes = {split: np.asarray(value).shape for split, value in splits.items()}
    for split, shape in shapes.items():
        if len(shape) != 4:
            return {"status": "error", "error_code": "HSI_ARRAY_SHAPE_INVALID", "message": f"{split} hsi must be [N, B, H, W]."}
        if shape[0] <= 0:
            return {"status": "error", "error_code": "EMPTY_SPLIT", "message": f"{split} split is empty."}
    band_counts = {shape[1] for shape in shapes.values()}
    spatial = {(shape[2], shape[3]) for shape in shapes.values()}
    if len(band_counts) != 1 or len(spatial) != 1:
        return {"status": "error", "error_code": "SPLIT_SHAPE_MISMATCH", "message": "Split band/spatial shapes must match."}
    return {"status": "valid", "shapes": {key: list(value) for key, value in shapes.items()}}


def save_prepared_dataset(
    arrays: dict[str, dict[str, Any]],
    output_dir: Path,
    dataset_id: str,
    dataset_family: str,
    source_path: Path,
    normalization: str,
    crop_size: int,
    patch_stride: int,
    license_note: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in _SPLITS:
        payload = dict(arrays[split])
        payload["hsi"] = np.asarray(payload["hsi"], dtype=np.float32)
        if "depth_indices" not in payload:
            payload["depth_indices"] = np.zeros(payload["hsi"].shape[0], dtype=np.int64)
        np.savez_compressed(output_dir / f"{split}.npz", **payload)
    train = np.asarray(arrays["train"]["hsi"])
    wavelengths = arrays["train"].get("wavelengths_nm")
    wavelengths_list = [float(item) for item in np.asarray(wavelengths).reshape(-1).tolist()] if wavelengths is not None else None
    wavelength_range = (float(wavelengths_list[0]), float(wavelengths_list[-1])) if wavelengths_list else (450.0, 700.0)
    spec = HSIDatasetSpec(
        dataset_id=dataset_id,
        dataset_name=f"{dataset_id}_hsi",
        source="synthetic" if dataset_family == "synthetic" else ("local" if dataset_family == "local_npz" else "public_placeholder"),
        dataset_family=dataset_family,  # type: ignore[arg-type]
        dataset_path=str(source_path),
        normalization=normalization,  # type: ignore[arg-type]
        crop_size=int(crop_size),
        patch_stride=int(patch_stride),
        spectral_bands=int(train.shape[1]),
        height=int(train.shape[2]),
        width=int(train.shape[3]),
        train_size=int(arrays["train"]["hsi"].shape[0]),
        val_size=int(arrays["val"]["hsi"].shape[0]),
        test_size=int(arrays["test"]["hsi"].shape[0]),
        wavelength_range_nm=wavelength_range,
        wavelengths_nm=wavelengths_list,
        data_license_note=license_note,
        preprocessing={"normalization": normalization, "crop_size": int(crop_size), "patch_stride": int(patch_stride)},
        metadata={"prepared_from": str(source_path), **(metadata or {})},
    )
    manifest = spec.model_dump(mode="json")
    manifest.update(
        {
            "status": "prepared",
            "dataset_id": dataset_id,
            "dataset_family": dataset_family,
            "source_path": str(source_path),
            "license_note": license_note,
            "band_count": int(train.shape[1]),
            "patch_count": int(sum(arrays[split]["hsi"].shape[0] for split in _SPLITS)),
            "output_dir": str(output_dir),
            "validation": validate_hsi_dataset_spec(spec),
        }
    )
    (output_dir / "dataset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    return manifest


def _ensure_bhw(cube: np.ndarray) -> np.ndarray:
    arr = np.asarray(cube, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected [B,H,W] or [H,W,B], got {arr.shape}")
    band_axis = infer_band_axis(arr)
    if band_axis == 2:
        arr = np.moveaxis(arr, 2, 0)
    return arr.astype(np.float32)


def _ensure_nbhw(hsi: np.ndarray) -> np.ndarray:
    arr = np.asarray(hsi, dtype=np.float32)
    if arr.ndim == 3:
        return arr[None]
    if arr.ndim != 4:
        raise ValueError(f"Expected hsi [N,B,H,W], got {arr.shape}")
    return arr


def _prepare_hsi_payload(
    hsi: np.ndarray,
    crop_size: int,
    stride: int,
    normalization: str,
    depth_indices: np.ndarray | None = None,
    wavelengths_nm: list[float] | np.ndarray | None = None,
    masks: np.ndarray | None = None,
) -> dict[str, Any]:
    arr = np.asarray(hsi, dtype=np.float32)
    if arr.ndim == 3:
        patches = crop_to_patches(arr, crop_size, stride)
        depths = np.zeros(patches.shape[0], dtype=np.int64)
    else:
        arr = _ensure_nbhw(arr)
        patch_chunks = []
        depth_chunks = []
        for idx, cube in enumerate(arr):
            patches = crop_to_patches(cube, crop_size, stride)
            patch_chunks.append(patches)
            depth_value = int(depth_indices[idx]) if depth_indices is not None and len(depth_indices) > idx else 0
            depth_chunks.append(np.full(patches.shape[0], depth_value, dtype=np.int64))
        patches = np.concatenate(patch_chunks, axis=0) if patch_chunks else np.empty((0, arr.shape[1], min(crop_size, arr.shape[2]), min(crop_size, arr.shape[3])), dtype=np.float32)
        depths = np.concatenate(depth_chunks, axis=0) if depth_chunks else np.empty((0,), dtype=np.int64)
    patches = normalize_hsi_cube(patches, normalization)
    payload: dict[str, Any] = {"hsi": patches, "depth_indices": depths}
    if wavelengths_nm is not None:
        payload["wavelengths_nm"] = np.asarray(wavelengths_nm, dtype=np.float32)
    if masks is not None:
        payload["masks"] = np.asarray(masks)
    return payload


def _payload_wavelengths(payload: dict[str, Any]) -> list[float] | None:
    wavelengths = payload.get("wavelengths_nm")
    if wavelengths is None:
        return None
    return [float(item) for item in np.asarray(wavelengths).reshape(-1).tolist()]


def _default_split_labels(count: int) -> np.ndarray:
    count = int(count)
    labels = np.empty(count, dtype=object)
    if count <= 1:
        labels[:] = "train"
        return labels.astype(str)
    train_end = max(1, int(round(count * 0.7)))
    val_end = max(train_end + 1, int(round(count * 0.85))) if count > 2 else train_end
    labels[:train_end] = "train"
    labels[train_end:val_end] = "val"
    labels[val_end:] = "test"
    if not np.any(labels == "test"):
        labels[-1] = "test"
    if count > 2 and not np.any(labels == "val"):
        labels[-2] = "val"
    return labels.astype(str)


def _ensure_nonempty_splits(arrays: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    for split in _SPLITS:
        if arrays[split]["hsi"].shape[0] == 0:
            fallback = next(payload for payload in arrays.values() if payload["hsi"].shape[0] > 0)
            replacement = {}
            for key, value in fallback.items():
                if isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] == fallback["hsi"].shape[0]:
                    replacement[key] = value[:1]
                else:
                    replacement[key] = value
            arrays[split] = replacement
    return arrays


def _starts(size: int, crop: int, stride: int) -> list[int]:
    if size <= crop:
        return [0]
    starts = list(range(0, size - crop + 1, stride))
    if starts[-1] != size - crop:
        starts.append(size - crop)
    return starts
