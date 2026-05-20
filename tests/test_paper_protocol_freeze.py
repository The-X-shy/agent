from optiresearch.reports.protocol_freeze import freeze_paper_protocol


def test_freeze_paper_protocol_exports_v01(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_REPORT_ROOT", str(tmp_path / "reports"))

    path = freeze_paper_protocol()
    text = path.read_text(encoding="utf-8")

    assert path.name == "paper_experiment_protocol_v0.1_freeze.md"
    assert "Evidence level definitions" in text
    assert "public_hsi_mock" in text
    assert "native_optimized" in text

