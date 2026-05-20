from optiresearch.runtime.hsi_matrix import run_hsi_matrix


def test_hsi_matrix_writes_outputs_and_ranks_best_by_reconstructor(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_DB_PATH", str(tmp_path / "memory.sqlite"))
    monkeypatch.setenv("OPTIRESEARCH_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("OPTIRESEARCH_HSI_ROOT", str(tmp_path / "hsi"))

    result = run_hsi_matrix(
        datasets=["synthetic"],
        backends=["mock_deeplens"],
        encoders=["conventional", "achromatic"],
        reconstructors=["optical_conditioned_linear", "tiny_cnn"],
        forward_modes=["depth_spectral_coded"],
        objective="Compare Phase 11 matrix",
    )

    assert result["matrix_id"]
    assert (tmp_path / "hsi" / "matrix" / result["matrix_id"] / "hsi_matrix_results.json").exists()
    assert result["summary"]["best_by_reconstructor"]["optical_conditioned_linear"]["encoder"] in {"conventional", "achromatic"}
    tiny_rows = [row for row in result["rows"] if row["reconstructor"] == "tiny_cnn"]
    assert tiny_rows
    assert all(row["status"] in {"succeeded", "skipped"} for row in tiny_rows)

