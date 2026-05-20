"""Local CAVE / ICVL dataset adapters.

No public dataset is downloaded by these adapters. They only scan user-provided
local directories and normalize found arrays into the common train/val/test NPZ
layout.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from optiresearch.hsi.public_datasets import LocalNPZHSIAdapter, _SPLITS, save_prepared_dataset


class _PublicSceneAdapter(LocalNPZHSIAdapter):
    dataset_id = "public"
    env_var = ""
    license_note = "Local public HSI dataset path. User is responsible for license compliance."

    def __init__(
        self,
        path: str | Path | None = None,
        dataset_id: str = "public",
        dataset_family: str = "custom",
        crop_size: int = 32,
        patch_stride: int = 32,
        normalization: str = "per_band",
    ) -> None:
        env_path = os.getenv(self.env_var) if self.env_var else None
        super().__init__(
            path or env_path,
            dataset_id=dataset_id,
            dataset_family=dataset_family,
            crop_size=crop_size,
            patch_stride=patch_stride,
            normalization=normalization,
        )

    def prepare(self, output_dir: Path) -> dict:
        output_dir = Path(output_dir)
        if self.path is None:
            return self._skip("DATASET_NOT_CONFIGURED", f"Set {self.env_var} to a local dataset directory.", output_dir)
        if not self.path.exists():
            return self._skip("DATASET_PATH_NOT_FOUND", f"Dataset path does not exist: {self.path}", output_dir)
        if super().available():
            return super().prepare(output_dir)
        files = self._scene_files()
        if not files:
            return self._skip("DATASET_FILES_NOT_FOUND", "No .npz, .npy, or .mat scenes found.", output_dir)
        arrays = []
        wavelengths = None
        for file_path in files:
            loaded = self._load_scene_file(file_path)
            if loaded.get("status") == "error":
                return self._skip(loaded["error_code"], loaded["message"], output_dir)
            arrays.append(loaded["cube"])
            wavelengths = loaded.get("wavelengths_nm") or wavelengths
        from optiresearch.hsi.public_datasets import _prepare_hsi_payload, _ensure_nonempty_splits, _default_split_labels

        prepared = _prepare_hsi_payload(
            np.asarray(arrays, dtype=np.float32),
            crop_size=self.crop_size,
            stride=self.patch_stride,
            normalization=self.normalization,
            wavelengths_nm=wavelengths,
        )
        labels = _default_split_labels(prepared["hsi"].shape[0])
        splits: dict[str, dict[str, Any]] = {}
        for split in _SPLITS:
            mask = labels == split
            splits[split] = {
                "hsi": prepared["hsi"][mask],
                "depth_indices": prepared["depth_indices"][mask],
            }
            if "wavelengths_nm" in prepared:
                splits[split]["wavelengths_nm"] = prepared["wavelengths_nm"]
        splits = _ensure_nonempty_splits(splits)
        manifest = save_prepared_dataset(
            splits,
            output_dir,
            dataset_id=self.dataset_id,
            dataset_family=self.dataset_family,
            source_path=self.path,
            normalization=self.normalization,
            crop_size=self.crop_size,
            patch_stride=self.patch_stride,
            license_note=self.license_note,
            metadata={"scene_files": [str(path) for path in files], "public_download_performed": False},
        )
        self.prepared_dir = output_dir
        return manifest

    def _scene_files(self) -> list[Path]:
        if self.path is None:
            return []
        if self.path.is_file():
            return [self.path]
        return sorted([*self.path.glob("*.npz"), *self.path.glob("*.npy"), *self.path.glob("*.mat")])

    def _load_scene_file(self, path: Path) -> dict[str, Any]:
        suffix = path.suffix.lower()
        if suffix == ".npy":
            return {"cube": np.load(path)}
        if suffix == ".npz":
            payload = np.load(path)
            data = {key: np.asarray(payload[key]) for key in payload.files}
            cube = data.get("cube")
            if cube is None:
                cube = data.get("hsi")
            if cube is None:
                cube = data[next(iter(data))]
            return {"cube": cube, "wavelengths_nm": data.get("wavelengths_nm")}
        if suffix == ".mat":
            try:
                from scipy.io import loadmat
            except Exception:
                return {"status": "error", "error_code": "SCIPY_NOT_AVAILABLE", "message": "scipy is required to read .mat files."}
            data = {key: value for key, value in loadmat(path).items() if not key.startswith("__")}
            for key in ("cube", "hsi", "rad", "reflectance"):
                if key in data:
                    return {"cube": data[key]}
            if data:
                return {"cube": next(iter(data.values()))}
        return {"status": "error", "error_code": "UNSUPPORTED_DATASET_FILE", "message": f"Unsupported scene file: {path}"}

    def _skip(self, code: str, message: str, output_dir: Path) -> dict[str, Any]:
        return {
            "status": "skipped",
            "dataset_id": self.dataset_id,
            "dataset_family": self.dataset_family,
            "error_code": code,
            "message": message,
            "source_path": str(self.path) if self.path else None,
            "output_dir": str(output_dir),
        }


class CAVEAdapter(_PublicSceneAdapter):
    dataset_id = "cave"
    env_var = "OPTIRESEARCH_CAVE_PATH"
    license_note = "CAVE dataset local copy. User is responsible for CAVE license and citation."

    def __init__(self, path: str | Path | None = None, crop_size: int = 32, patch_stride: int = 32, normalization: str = "per_band") -> None:
        super().__init__(path, dataset_id="cave", dataset_family="cave", crop_size=crop_size, patch_stride=patch_stride, normalization=normalization)


class ICVLAdapter(_PublicSceneAdapter):
    dataset_id = "icvl"
    env_var = "OPTIRESEARCH_ICVL_PATH"
    license_note = "ICVL dataset local copy. User is responsible for ICVL license and citation."

    def __init__(self, path: str | Path | None = None, crop_size: int = 32, patch_stride: int = 32, normalization: str = "per_band") -> None:
        super().__init__(path, dataset_id="icvl", dataset_family="icvl", crop_size=crop_size, patch_stride=patch_stride, normalization=normalization)
