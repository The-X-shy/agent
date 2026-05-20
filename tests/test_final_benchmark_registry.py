"""Test FinalBenchmarkRegistry."""
from pathlib import Path
from optiresearch.runtime.final_benchmark import FinalBenchmarkRegistry


def test_registry_lists_five_benchmark_groups(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_BENCHMARK_ROOT", str(tmp_path / "benchmarks"))
    monkeypatch.setenv("OPTIRESEARCH_REPORT_ROOT", str(tmp_path / "reports"))
    monkeypatch.setenv("OPTIRESEARCH_HSI_ROOT", str(tmp_path / "hsi"))

    registry = FinalBenchmarkRegistry()
    benchmarks = registry.list_benchmarks()

    assert isinstance(benchmarks, list)
    groups = {b["group"] for b in benchmarks}
    assert "A_system" in groups
    assert "B_optical_backend" in groups
    assert "C_hsi_synthetic" in groups
    assert "D_public_local_hsi" in groups
    assert "E_evidence" in groups


def test_registry_validates_required_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_BENCHMARK_ROOT", str(tmp_path / "benchmarks"))
    monkeypatch.setenv("OPTIRESEARCH_REPORT_ROOT", str(tmp_path / "reports"))
    monkeypatch.setenv("OPTIRESEARCH_HSI_ROOT", str(tmp_path / "hsi"))

    registry = FinalBenchmarkRegistry()
    result = registry.validate_required_artifacts()

    assert isinstance(result, dict)
    assert "status" in result
    assert "missing" in result
    assert "present" in result


def test_registry_collects_results(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_BENCHMARK_ROOT", str(tmp_path / "benchmarks"))
    monkeypatch.setenv("OPTIRESEARCH_REPORT_ROOT", str(tmp_path / "reports"))
    monkeypatch.setenv("OPTIRESEARCH_HSI_ROOT", str(tmp_path / "hsi"))

    registry = FinalBenchmarkRegistry()
    results = registry.collect_results()

    assert isinstance(results, dict)
    for group in ["A_system", "B_optical_backend", "C_hsi_synthetic", "D_public_local_hsi", "E_evidence"]:
        assert group in results


def test_registry_exports_summary(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_BENCHMARK_ROOT", str(tmp_path / "benchmarks"))
    monkeypatch.setenv("OPTIRESEARCH_REPORT_ROOT", str(tmp_path / "reports"))
    monkeypatch.setenv("OPTIRESEARCH_HSI_ROOT", str(tmp_path / "hsi"))

    output_dir = tmp_path / "final_benchmark"
    registry = FinalBenchmarkRegistry()
    exported = registry.export_summary(output_dir)

    assert "summary_json" in exported
    assert "summary_md" in exported
    assert "artifact_inventory" in exported
    assert Path(exported["summary_json"]).exists()
    assert Path(exported["summary_md"]).exists()
    assert Path(exported["artifact_inventory"]).exists()
