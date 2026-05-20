"""Test final paper package export."""
import json
from pathlib import Path
from optiresearch.reports.final_package import export_final_paper_package


def test_export_final_paper_package_creates_all_files(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_REPORT_ROOT", str(tmp_path / "workspace" / "reports"))
    monkeypatch.setenv("OPTIRESEARCH_HSI_ROOT", str(tmp_path / "hsi"))
    monkeypatch.setenv("OPTIRESEARCH_BENCHMARK_ROOT", str(tmp_path / "benchmarks"))
    monkeypatch.setenv("OPTIRESEARCH_DB_PATH", str(tmp_path / "test.db"))

    output_dir = tmp_path / "final_paper_package"
    result = export_final_paper_package(output_dir)

    assert "package_dir" in result
    assert "manifest_path" in result

    pkg = Path(result["package_dir"])
    assert (pkg / "README.md").exists()
    assert (pkg / "claim_boundary.md").exists()
    assert (pkg / "evidence_distribution.md").exists()
    assert (pkg / "artifact_inventory.json").exists()
    assert (pkg / "reproducibility_manifest.json").exists()
    assert (pkg / "paper_tables").is_dir()
    assert (pkg / "phase_reports").is_dir()


def test_reproducibility_manifest_has_required_fields(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_REPORT_ROOT", str(tmp_path / "workspace" / "reports"))
    monkeypatch.setenv("OPTIRESEARCH_HSI_ROOT", str(tmp_path / "hsi"))
    monkeypatch.setenv("OPTIRESEARCH_DB_PATH", str(tmp_path / "test.db"))

    output_dir = tmp_path / "final_paper_package"
    result = export_final_paper_package(output_dir)

    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))

    for key in ["code_version", "python_version", "package_summary", "workspace_paths",
                "dataset_availability", "deeplens_availability", "llm_provider_availability",
                "timestamp", "limitations"]:
        assert key in manifest, f"Missing manifest key: {key}"


def test_final_package_readme_is_complete(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_REPORT_ROOT", str(tmp_path / "workspace" / "reports"))
    monkeypatch.setenv("OPTIRESEARCH_HSI_ROOT", str(tmp_path / "hsi"))
    monkeypatch.setenv("OPTIRESEARCH_DB_PATH", str(tmp_path / "test.db"))

    output_dir = tmp_path / "final_paper_package"
    result = export_final_paper_package(output_dir)

    readme = (Path(result["package_dir"]) / "README.md").read_text(encoding="utf-8")
    assert "Final Paper Package" in readme
    assert "Reproducibility" in readme
    assert "Limitations" in readme


def test_phase_reports_copied(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_REPORT_ROOT", str(tmp_path / "workspace" / "reports"))
    monkeypatch.setenv("OPTIRESEARCH_HSI_ROOT", str(tmp_path / "hsi"))
    monkeypatch.setenv("OPTIRESEARCH_DB_PATH", str(tmp_path / "test.db"))

    reports_dir = tmp_path / "workspace" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    for name in ["phase10_optical_sensitive_hsi_report.md",
                 "phase11_hsi_network_dataset_report.md",
                 "phase12_public_hsi_deeplens_protocol_report.md"]:
        (reports_dir / name).write_text(f"# {name}", encoding="utf-8")

    output_dir = tmp_path / "final_paper_package"
    result = export_final_paper_package(output_dir)

    phase_dir = Path(result["package_dir"]) / "phase_reports"
    assert phase_dir.exists()
    files = list(phase_dir.iterdir())
    assert len(files) >= 1
