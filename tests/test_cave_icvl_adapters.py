import numpy as np

from optiresearch.hsi.cave_icvl import CAVEAdapter, ICVLAdapter


def test_cave_adapter_prepares_fake_npz_scene(tmp_path, monkeypatch):
    source = tmp_path / "cave"
    source.mkdir()
    np.savez_compressed(source / "scene.npz", cube=np.ones((6, 8, 8), dtype=np.float32), wavelengths_nm=np.linspace(400.0, 700.0, 6))
    monkeypatch.setenv("OPTIRESEARCH_CAVE_PATH", str(source))

    adapter = CAVEAdapter(crop_size=4, patch_stride=4)
    result = adapter.prepare(tmp_path / "prepared")

    assert result["status"] == "prepared"
    assert result["dataset_family"] == "cave"
    assert result["band_count"] == 6
    assert (tmp_path / "prepared" / "dataset_manifest.json").exists()


def test_icvl_adapter_prepares_fake_npy_scene(tmp_path, monkeypatch):
    source = tmp_path / "icvl"
    source.mkdir()
    np.save(source / "scene.npy", np.ones((8, 8, 4), dtype=np.float32))
    monkeypatch.setenv("OPTIRESEARCH_ICVL_PATH", str(source))

    adapter = ICVLAdapter(crop_size=4, patch_stride=4)
    result = adapter.prepare(tmp_path / "prepared")

    assert result["status"] == "prepared"
    assert result["dataset_family"] == "icvl"
    assert result["band_count"] == 4


def test_cave_adapter_not_configured_returns_structured_error(monkeypatch, tmp_path):
    monkeypatch.delenv("OPTIRESEARCH_CAVE_PATH", raising=False)

    result = CAVEAdapter().prepare(tmp_path / "prepared")

    assert result["status"] == "skipped"
    assert result["error_code"] == "DATASET_NOT_CONFIGURED"

