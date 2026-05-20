from optiresearch.runtime.hsi_public_matrix import run_public_hsi_matrix


def test_public_hsi_matrix_returns_structured_skip_when_dataset_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_HSI_ROOT", str(tmp_path / "hsi"))

    result = run_public_hsi_matrix(dataset="local_npz", dataset_path=str(tmp_path / "missing"), backend="mock_deeplens")

    assert result["status"] == "skipped"
    assert result["error_code"] == "DATASET_PATH_NOT_FOUND"
    assert result["summary"]["skipped"] >= 1

