"""Test Phase 16 report generation."""
from optiresearch.reports.phase16 import export_phase16_report


def test_export_phase16_report_creates_file(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_REPORT_ROOT", str(tmp_path / "reports"))
    monkeypatch.setenv("OPTIRESEARCH_HSI_ROOT", str(tmp_path / "hsi"))
    monkeypatch.setenv("OPTIRESEARCH_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("OPTIRESEARCH_ARTIFACT_ROOT", str(tmp_path / "artifacts"))

    path = export_phase16_report()
    assert path.exists()
    assert path.name == "phase16_deeplens_backed_codesign_report.md"


def test_phase16_report_contains_required_sections(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_REPORT_ROOT", str(tmp_path / "reports"))
    monkeypatch.setenv("OPTIRESEARCH_HSI_ROOT", str(tmp_path / "hsi"))
    monkeypatch.setenv("OPTIRESEARCH_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("OPTIRESEARCH_ARTIFACT_ROOT", str(tmp_path / "artifacts"))

    path = export_phase16_report()
    text = path.read_text(encoding="utf-8")

    required = [
        "Phase 16",
        "DeepLens-Backed",
        "Phase 15 Limitation",
        "PSF Mapping",
        "Supported and Unsupported",
        "Evidence Level",
        "What Is Validated",
        "What Is NOT Validated",
        "Requirements for Differentiable",
        "Phase 17",
    ]
    for section in required:
        assert section.lower() in text.lower(), f"Missing: {section}"
