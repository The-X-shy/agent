import json

import numpy as np

from optiresearch.hsi.public_datasets import (
    CAVEPlaceholderAdapter,
    ICVLPlaceholderAdapter,
    LocalNPZHSIAdapter,
    SyntheticAdapter,
    get_hsi_dataset_adapter,
    list_hsi_dataset_adapters,
)


def _write_local_split(path, name, value):
    hsi = np.full((2, 4, 8, 8), value, dtype=np.float32)
    depth_indices = np.array([0, 1], dtype=np.int64)
    wavelengths_nm = np.linspace(450.0, 700.0, 4, dtype=np.float32)
    np.savez_compressed(path / f"{name}.npz", hsi=hsi, depth_indices=depth_indices, wavelengths_nm=wavelengths_nm)


def test_synthetic_adapter_prepares_manifest_and_splits(tmp_path):
    adapter = SyntheticAdapter()

    result = adapter.prepare(tmp_path / "synthetic")
    train = adapter.load_split("train")

    assert adapter.available()
    assert result["status"] == "prepared"
    assert result["dataset_id"] == "synthetic"
    assert (tmp_path / "synthetic" / "dataset_manifest.json").exists()
    assert train["hsi"].ndim == 4
    assert json.loads((tmp_path / "synthetic" / "dataset_manifest.json").read_text())["dataset_family"] == "synthetic"


def test_local_npz_adapter_prepares_from_path_and_preserves_optional_fields(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for split, value in {"train": 0.2, "val": 0.4, "test": 0.6}.items():
        _write_local_split(source, split, value)

    adapter = LocalNPZHSIAdapter(source)
    result = adapter.prepare(tmp_path / "prepared")
    test = adapter.load_split("test")

    assert adapter.available()
    assert result["status"] == "prepared"
    assert result["dataset_id"] == "local_npz"
    assert test["hsi"].shape == (2, 4, 8, 8)
    assert "wavelengths_nm" in test
    assert (tmp_path / "prepared" / "train.npz").exists()


def test_local_npz_missing_path_returns_structured_error(tmp_path):
    adapter = LocalNPZHSIAdapter(tmp_path / "missing")

    result = adapter.prepare(tmp_path / "prepared")

    assert not adapter.available()
    assert result["status"] == "error"
    assert result["error_code"] == "DATASET_PATH_NOT_FOUND"


def test_public_placeholders_do_not_download_and_report_not_configured(monkeypatch, tmp_path):
    monkeypatch.delenv("OPTIRESEARCH_CAVE_PATH", raising=False)
    monkeypatch.delenv("OPTIRESEARCH_ICVL_PATH", raising=False)

    for adapter in (CAVEPlaceholderAdapter(), ICVLPlaceholderAdapter()):
        result = adapter.prepare(tmp_path / adapter.dataset_id)
        assert not adapter.available()
        assert result["status"] == "error"
        assert result["error_code"] == "DATASET_NOT_CONFIGURED"
        assert "expected_structure" in result


def test_dataset_registry_lists_and_resolves_adapters():
    listed = list_hsi_dataset_adapters()

    assert {"synthetic", "local_npz", "cave", "icvl"}.issubset(set(listed))
    assert isinstance(get_hsi_dataset_adapter("synthetic"), SyntheticAdapter)

