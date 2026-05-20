from optiresearch.reports.phase8 import export_phase8_report


def test_phase8_report_exports(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_REPORT_ROOT", str(tmp_path / "reports"))
    monkeypatch.setenv("OPTIRESEARCH_BASELINE_ROOT", str(tmp_path / "baselines"))

    path = export_phase8_report()

    text = path.read_text(encoding="utf-8")
    assert path.name == "phase8_deeplens_semi_native_report.md"
    assert "DeepLens API probe summary" in text
    assert "Adapter-proxy vs semi-native distinction" in text
    assert "Optimization readiness" in text
