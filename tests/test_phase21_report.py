"""Tests for Phase 21 report generation."""


def test_phase21_report_generates_without_crashing(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_REPORT_ROOT", str(tmp_path / "workspace" / "reports"))
    from optiresearch.reports.phase21 import export_phase21_report

    path = export_phase21_report()
    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert "Phase 21" in content
    assert "Reconstruction" in content
    assert "ClaimEvidence" in content


def test_phase21_report_includes_all_sections(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_REPORT_ROOT", str(tmp_path / "workspace" / "reports"))
    from optiresearch.reports.phase21 import export_phase21_report

    path = export_phase21_report()
    content = path.read_text(encoding="utf-8")

    for section in ["Objective", "Phase 20 Recap", "Reconstruction Module",
                    "ClaimEvidence", "What Is Validated", "What Is Still Proxy",
                    "Requirements"]:
        assert section in content, f"Missing: {section}"
