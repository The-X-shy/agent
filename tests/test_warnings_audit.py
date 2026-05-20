"""Test warnings audit module."""
from pathlib import Path
from optiresearch.reports.warnings_audit import classify_warnings, WarningsAudit


def test_classify_warnings_categorizes_deprecation():
    lines = [
        "test_file.py:10: DeprecationWarning: some_func is deprecated",
        "test_file.py:20: PytestRemovedIn8Warning: some feature",
        "test_file.py:30: UserWarning: optional dependency torch not available",
        "test_file.py:40: Skipped: OPTIRESEARCH_ENABLE_REAL_DEEPLENS_TESTS",
        "test_file.py:50: RuntimeWarning: invalid value encountered in divide",
        "test_file.py:60: FileNotFoundError: /some/path",
        "test_file.py:70: SomeOtherWarning: unknown thing",
    ]
    result = classify_warnings(lines)

    assert "deprecation" in result
    assert "optional_dependency" in result
    assert "test_skip" in result
    assert "numerical" in result
    assert "file_path" in result
    assert "unknown" in result

    assert result["deprecation"] > 0
    assert result["optional_dependency"] > 0
    assert result["test_skip"] > 0
    assert result["numerical"] > 0


def test_classify_warnings_empty_input():
    result = classify_warnings([])
    for v in result.values():
        assert v == 0


def test_warnings_audit_generates_markdown(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_REPORT_ROOT", str(tmp_path / "reports"))

    audit = WarningsAudit()
    lines = [
        "test_a.py:1: DeprecationWarning: old API",
        "test_b.py:2: UserWarning: optional torch missing",
    ]
    md = audit.generate_report(lines)

    assert "Warnings Audit" in md
    assert "Deprecation" in md
    assert "Optional Dependency" in md


def test_warnings_audit_export_writes_file(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_REPORT_ROOT", str(tmp_path / "reports"))

    audit = WarningsAudit()
    path = audit.export_report(["test.py:1: DeprecationWarning: x"], output_dir=tmp_path / "reports")

    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert "Warnings Audit" in content
