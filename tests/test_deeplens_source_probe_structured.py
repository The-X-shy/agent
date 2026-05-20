"""Test structured DeepLens source probe output."""
import json
import os
from pathlib import Path
from optiresearch.adapters.deeplens import DeepLensAdapter


def test_probe_output_has_all_keys(monkeypatch, tmp_path):
    repo_path = os.getenv("DEEPLENS_REPO_PATH", str(tmp_path / "missing_deeplens_repo"))
    monkeypatch.setenv("DEEPLENS_REPO_PATH", repo_path)
    monkeypatch.setenv("OPTIRESEARCH_REPORT_ROOT", str(tmp_path / "reports"))

    adapter = DeepLensAdapter()
    env = adapter.validate_environment()

    required_keys = [
        "available", "import_path", "repo_path", "is_source_checkout",
        "deeplens_version", "available_modules", "available_classes",
        "missing_modules", "capabilities", "source_repo",
    ]
    for key in required_keys:
        assert key in env, f"Missing key: {key}"


def test_probe_json_exportable(monkeypatch, tmp_path):
    repo_path = os.getenv("DEEPLENS_REPO_PATH", str(tmp_path / "missing_deeplens_repo"))
    monkeypatch.setenv("DEEPLENS_REPO_PATH", repo_path)
    monkeypatch.setenv("OPTIRESEARCH_REPORT_ROOT", str(tmp_path / "reports"))

    adapter = DeepLensAdapter()
    env = adapter.validate_environment()

    json_path = tmp_path / "reports" / "deeplens_source_probe.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(env, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    assert json_path.exists()
    reloaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert reloaded["available"] == env["available"]


def test_unavailable_probe_is_structured(monkeypatch):
    monkeypatch.delenv("DEEPLENS_REPO_PATH", raising=False)
    # Simulate no deeplens by not setting path
    adapter = DeepLensAdapter(deeplens_module=None)
    adapter._deeplens = None
    adapter._import_error = "Simulated import failure"
    env = adapter.validate_environment()

    assert env["available"] is False
    assert env["error_code"] is not None
    assert env["is_source_checkout"] is False
