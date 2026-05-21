"""Tests for Phase 20 report generation."""


def test_phase20_report_generates_without_crashing(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_REPORT_ROOT", str(tmp_path / "workspace" / "reports"))
    from optiresearch.reports.phase20 import export_phase20_report

    path = export_phase20_report()
    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert "Phase 20" in content
    assert "Differentiable HSI Proxy" in content
    assert "ClaimEvidence Decision" in content


def test_phase20_report_includes_all_sections(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_REPORT_ROOT", str(tmp_path / "workspace" / "reports"))
    from optiresearch.reports.phase20 import export_phase20_report

    path = export_phase20_report()
    content = path.read_text(encoding="utf-8")

    required_sections = [
        "Objective",
        "Phase 19B Findings",
        "Differentiable HSI Proxy Design",
        "Fresnel Native HSI Proxy",
        "Binary2Phase Native HSI Proxy",
        "Remote WSL",
        "ClaimEvidence Decision",
        "What Is Validated",
        "What Is NOT Validated",
        "Requirements",
    ]
    for section in required_sections:
        assert section in content, f"Missing section: {section}"
