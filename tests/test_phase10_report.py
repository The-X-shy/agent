"""Test Phase 10 report generation."""

from optiresearch.reports.phase10 import export_phase10_report


def test_export_phase10_report(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_REPORT_ROOT", str(tmp_path / "reports"))
    path = export_phase10_report()
    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert "Phase 10" in content
    assert "optical-sensitive" in content.lower()
    assert "mixed_materials" in content
    assert "depth_spectral_coded" in content
    assert "optical_conditioned_linear" in content


def test_export_phase10_report_produces_valid_markdown(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_REPORT_ROOT", str(tmp_path / "reports"))
    path = export_phase10_report()
    lines = path.read_text(encoding="utf-8").split("\n")
    assert lines[0].startswith("# ")
    assert any("## " in line for line in lines)
