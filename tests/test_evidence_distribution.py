"""Test evidence distribution computation."""
from optiresearch.reports.evidence_distribution import compute_evidence_distribution


def test_compute_evidence_distribution_returns_expected_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_REPORT_ROOT", str(tmp_path / "reports"))
    monkeypatch.setenv("OPTIRESEARCH_HSI_ROOT", str(tmp_path / "hsi"))
    monkeypatch.setenv("OPTIRESEARCH_DB_PATH", str(tmp_path / "test.db"))

    result = compute_evidence_distribution()

    for key in ["count_by_level", "status_counts", "artifact_coverage", "missing_evidence_warnings"]:
        assert key in result, f"Missing key: {key}"

    assert "mock" in result["count_by_level"]
    assert "synthetic_hsi" in result["count_by_level"]
    assert "supported" in result["status_counts"]


def test_evidence_distribution_counts_are_non_negative(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_REPORT_ROOT", str(tmp_path / "reports"))
    monkeypatch.setenv("OPTIRESEARCH_DB_PATH", str(tmp_path / "test.db"))

    result = compute_evidence_distribution()

    for level, count in result["count_by_level"].items():
        assert count >= 0, f"Negative count for {level}: {count}"
    for status, count in result["status_counts"].items():
        assert count >= 0, f"Negative count for {status}: {count}"


def test_evidence_distribution_missing_warnings_is_list(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_REPORT_ROOT", str(tmp_path / "reports"))
    monkeypatch.setenv("OPTIRESEARCH_DB_PATH", str(tmp_path / "test.db"))

    result = compute_evidence_distribution()
    assert isinstance(result["missing_evidence_warnings"], list)
