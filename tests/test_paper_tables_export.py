"""Test paper-ready table export."""
import json
from pathlib import Path
from optiresearch.reports.paper_tables import export_paper_tables


def test_export_paper_tables_creates_all_outputs(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_REPORT_ROOT", str(tmp_path / "reports"))
    monkeypatch.setenv("OPTIRESEARCH_HSI_ROOT", str(tmp_path / "hsi"))
    monkeypatch.setenv("OPTIRESEARCH_BENCHMARK_ROOT", str(tmp_path / "benchmarks"))

    result = export_paper_tables()

    assert "markdown_dir" in result
    assert "csv_dir" in result
    assert "json_path" in result
    assert "all_md" in result

    md_dir = Path(result["markdown_dir"])
    assert md_dir.exists()

    assert Path(result["json_path"]).exists()
    assert Path(result["all_md"]).exists()


def test_export_paper_tables_all_tables_md_is_complete(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_REPORT_ROOT", str(tmp_path / "reports"))
    monkeypatch.setenv("OPTIRESEARCH_HSI_ROOT", str(tmp_path / "hsi"))

    result = export_paper_tables()
    all_md = Path(result["all_md"]).read_text(encoding="utf-8")

    assert "Table 1" in all_md
    assert "Table 2" in all_md
    assert "Table 10" in all_md
    assert "System Components" in all_md
    assert "Evidence Levels" in all_md
    assert "Claim Whitelist" in all_md


def test_export_paper_tables_json_is_valid(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_REPORT_ROOT", str(tmp_path / "reports"))
    monkeypatch.setenv("OPTIRESEARCH_HSI_ROOT", str(tmp_path / "hsi"))

    result = export_paper_tables()
    data = json.loads(Path(result["json_path"]).read_text(encoding="utf-8"))

    assert "tables" in data
    assert len(data["tables"]) == 10
