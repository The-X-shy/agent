from optiresearch.reports.phase9 import export_phase9_report


def test_phase9_report_exports(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_REPORT_ROOT", str(tmp_path / "reports"))
    monkeypatch.setenv("OPTIRESEARCH_HSI_BASELINE_ROOT", str(tmp_path / "hsi_baselines"))

    path = export_phase9_report()

    text = path.read_text(encoding="utf-8")
    assert path.name == "phase9_hsi_reconstruction_report.md"
    assert "HSI dataset spec" in text
    assert "Reconstruction baseline" in text
    assert "What is not validated" in text
