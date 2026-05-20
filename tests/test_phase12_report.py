from optiresearch.reports.phase12 import export_phase12_report


def test_phase12_report_exports_required_sections(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_REPORT_ROOT", str(tmp_path / "reports"))
    monkeypatch.setenv("OPTIRESEARCH_HSI_ROOT", str(tmp_path / "hsi"))

    path = export_phase12_report()
    text = path.read_text(encoding="utf-8")

    assert path.name == "phase12_public_hsi_deeplens_protocol_report.md"
    assert "DeepLens wavelength-aware PSF contract" in text
    assert "Public HSI matrix status" in text
    assert "Frozen paper experiment protocol" in text
