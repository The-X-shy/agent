import numpy as np

from optiresearch.runtime.hsi_public_matrix import run_public_hsi_matrix


def test_public_hsi_matrix_runs_with_fake_local_npz(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_DB_PATH", str(tmp_path / "memory.sqlite"))
    monkeypatch.setenv("OPTIRESEARCH_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("OPTIRESEARCH_HSI_ROOT", str(tmp_path / "hsi"))
    source = tmp_path / "local"
    source.mkdir()
    for split in ("train", "val", "test"):
        np.savez_compressed(source / f"{split}.npz", hsi=np.random.default_rng(3).random((1, 4, 8, 8), dtype=np.float32), wavelengths_nm=np.linspace(450, 650, 4))

    result = run_public_hsi_matrix(
        dataset="local_npz",
        dataset_path=str(source),
        backend="mock_deeplens",
        encoders=["conventional"],
        reconstructors=["optical_conditioned_linear"],
        forward_modes=["depth_spectral_coded"],
    )

    assert result["status"] == "succeeded"
    assert result["rows"][0]["dataset_family"] == "local_npz"
    assert result["rows"][0]["evidence_level"] == "public_hsi_mock"
    assert (tmp_path / "hsi" / "public_matrix" / result["matrix_id"] / "public_hsi_matrix_results.json").exists()

