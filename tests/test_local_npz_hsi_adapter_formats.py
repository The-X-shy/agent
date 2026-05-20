import json

import numpy as np

from optiresearch.hsi.public_datasets import (
    LocalNPZHSIAdapter,
    crop_to_patches,
    infer_band_axis,
    normalize_hsi_cube,
    validate_dataset_shapes,
)


def test_local_npz_format_a_split_files_with_preprocessing(tmp_path):
    source = tmp_path / "format_a"
    source.mkdir()
    wavelengths = np.linspace(450.0, 700.0, 4, dtype=np.float32)
    for split, value in {"train": 1.0, "val": 2.0, "test": 3.0}.items():
        np.savez_compressed(
            source / f"{split}.npz",
            hsi=np.full((2, 4, 8, 8), value, dtype=np.float32),
            depth_indices=np.array([0, 1], dtype=np.int64),
            wavelengths_nm=wavelengths,
        )

    adapter = LocalNPZHSIAdapter(source, crop_size=4, patch_stride=4, normalization="global")
    result = adapter.prepare(tmp_path / "prepared_a")

    train = adapter.load_split("train")
    manifest = json.loads((tmp_path / "prepared_a" / "dataset_manifest.json").read_text())
    assert result["status"] == "prepared"
    assert train["hsi"].shape[1:] == (4, 4, 4)
    assert float(train["hsi"].max()) <= 1.0
    assert manifest["preprocessing"]["normalization"] == "global"
    assert manifest["patch_count"] == result["patch_count"]


def test_local_npz_format_b_dataset_npz_with_split_array(tmp_path):
    source = tmp_path / "format_b"
    source.mkdir()
    hsi = np.random.default_rng(0).random((5, 3, 6, 6), dtype=np.float32)
    split = np.array(["train", "train", "val", "test", "test"])
    np.savez_compressed(source / "dataset.npz", hsi=hsi, split=split, wavelengths_nm=np.array([500.0, 600.0, 700.0]))

    adapter = LocalNPZHSIAdapter(source, crop_size=6, patch_stride=6, normalization="none")
    result = adapter.prepare(tmp_path / "prepared_b")

    assert result["status"] == "prepared"
    assert adapter.load_split("train")["hsi"].shape[0] == 2
    assert adapter.load_split("val")["hsi"].shape[0] == 1
    assert adapter.load_split("test")["hsi"].shape[0] == 2


def test_local_npz_format_c_single_cube_hwb_auto_patch(tmp_path):
    source = tmp_path / "format_c"
    source.mkdir()
    cube_hwb = np.random.default_rng(1).random((8, 8, 5), dtype=np.float32)
    np.savez_compressed(source / "scene.npz", cube=cube_hwb, wavelengths_nm=np.linspace(450.0, 650.0, 5))

    adapter = LocalNPZHSIAdapter(source, crop_size=4, patch_stride=4, normalization="per_band")
    result = adapter.prepare(tmp_path / "prepared_c")

    assert result["status"] == "prepared"
    assert result["band_count"] == 5
    assert result["patch_count"] == 4
    assert adapter.load_split("train")["hsi"].shape[1:] == (5, 4, 4)


def test_local_npz_helpers_validate_shapes_and_band_axis():
    cube_bhw = np.ones((4, 8, 8), dtype=np.float32)
    cube_hwb = np.ones((8, 8, 4), dtype=np.float32)

    assert infer_band_axis(cube_bhw) == 0
    assert infer_band_axis(cube_hwb) == 2
    assert crop_to_patches(cube_bhw, crop_size=4, stride=4).shape == (4, 4, 4, 4)
    assert normalize_hsi_cube(cube_bhw, "none").shape == cube_bhw.shape
    assert validate_dataset_shapes({"train": np.ones((1, 4, 8, 8)), "val": np.ones((1, 4, 8, 8)), "test": np.ones((1, 4, 8, 8))})["status"] == "valid"

