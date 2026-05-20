"""Test Phase 13 final benchmark report."""
from optiresearch.reports.phase13 import export_phase13_report


def test_export_phase13_report_creates_file(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_REPORT_ROOT", str(tmp_path / "reports"))
    monkeypatch.setenv("OPTIRESEARCH_HSI_ROOT", str(tmp_path / "hsi"))
    monkeypatch.setenv("OPTIRESEARCH_BENCHMARK_ROOT", str(tmp_path / "benchmarks"))
    monkeypatch.setenv("OPTIRESEARCH_DB_PATH", str(tmp_path / "test.db"))

    path = export_phase13_report()
    assert path.exists()
    assert path.name == "phase13_final_benchmark_report.md"


def test_phase13_report_contains_required_sections(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_REPORT_ROOT", str(tmp_path / "reports"))
    monkeypatch.setenv("OPTIRESEARCH_HSI_ROOT", str(tmp_path / "hsi"))
    monkeypatch.setenv("OPTIRESEARCH_BENCHMARK_ROOT", str(tmp_path / "benchmarks"))
    monkeypatch.setenv("OPTIRESEARCH_DB_PATH", str(tmp_path / "test.db"))

    path = export_phase13_report()
    text = path.read_text(encoding="utf-8")

    required_sections = [
        "Phase 13",
        "Objective",
        "System maturity",
        "Final benchmark",
        "Paper-ready tables",
        "Claim boundary",
        "Evidence distribution",
        "ready for paper writing",
        "requires native optimization",
        "Phase 14",
    ]
    for section in required_sections:
        assert section.lower() in text.lower(), f"Missing section: {section}"


def test_phase13_report_is_valid_markdown(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_REPORT_ROOT", str(tmp_path / "reports"))
    monkeypatch.setenv("OPTIRESEARCH_HSI_ROOT", str(tmp_path / "hsi"))
    monkeypatch.setenv("OPTIRESEARCH_DB_PATH", str(tmp_path / "test.db"))

    path = export_phase13_report()
    lines = path.read_text(encoding="utf-8").split("\n")
    assert lines[0].startswith("# ")
    assert any("## " in line for line in lines)


def test_phase13_report_lists_all_benchmark_groups(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_REPORT_ROOT", str(tmp_path / "reports"))
    monkeypatch.setenv("OPTIRESEARCH_HSI_ROOT", str(tmp_path / "hsi"))
    monkeypatch.setenv("OPTIRESEARCH_DB_PATH", str(tmp_path / "test.db"))

    path = export_phase13_report()
    text = path.read_text(encoding="utf-8")

    for group_label in ["System Benchmark", "Optical Backend Benchmark", "HSI Synthetic Benchmark", "Public/Local HSI Benchmark", "Evidence Benchmark"]:
        assert group_label in text, f"Missing benchmark group label: {group_label}"
