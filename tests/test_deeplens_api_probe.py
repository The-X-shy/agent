from optiresearch.adapters.deeplens_api_probe import export_deeplens_api_probe, probe_deeplens_api


def test_deeplens_api_probe_never_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_REPORT_ROOT", str(tmp_path / "reports"))

    result = probe_deeplens_api()
    paths = export_deeplens_api_probe()

    assert "available" in result
    assert "classes_discovered" in result
    assert "errors" in result
    assert paths["json"].exists()
    assert paths["markdown"].exists()
