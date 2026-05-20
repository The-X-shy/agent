from optiresearch.reports.phase11 import export_phase11_report


def test_phase11_report_exports_required_sections(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_HSI_ROOT", str(tmp_path / "hsi"))
    monkeypatch.setenv("OPTIRESEARCH_REPORT_ROOT", str(tmp_path / "reports"))
    matrix_dir = tmp_path / "hsi" / "matrix" / "matrix_report"
    matrix_dir.mkdir(parents=True)
    (matrix_dir / "hsi_matrix_summary.json").write_text(
        '{"matrix_id":"matrix_report","best_by_reconstructor":{"optical_conditioned_linear":{"encoder":"achromatic"}}}',
        encoding="utf-8",
    )

    path = export_phase11_report()
    text = path.read_text(encoding="utf-8")

    assert path.name == "phase11_hsi_network_dataset_report.md"
    assert "Objective" in text
    assert "Dataset adapter status" in text
    assert "What is not validated" in text
    assert "native DeepLens" in text

