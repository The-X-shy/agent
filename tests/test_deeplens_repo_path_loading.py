"""Test DEEPLENS_REPO_PATH source loading."""
import os
from optiresearch.adapters.deeplens import DeepLensAdapter


def _configured_repo_path(tmp_path):
    return os.getenv("DEEPLENS_REPO_PATH", str(tmp_path / "missing_deeplens_repo"))


def test_repo_path_env_var_detected(monkeypatch, tmp_path):
    monkeypatch.setenv("DEEPLENS_REPO_PATH", _configured_repo_path(tmp_path))
    adapter = DeepLensAdapter()
    env = adapter.validate_environment()

    assert "repo_path" in env
    assert "is_source_checkout" in env
    assert "import_path" in env
    assert "available_modules" in env
    assert "available_classes" in env
    assert "missing_modules" in env


def test_available_modules_structure(monkeypatch, tmp_path):
    monkeypatch.setenv("DEEPLENS_REPO_PATH", _configured_repo_path(tmp_path))
    adapter = DeepLensAdapter()
    env = adapter.validate_environment()

    if env["available"]:
        modules = env["available_modules"]
        assert isinstance(modules, dict)
        assert "paraxiallens" in modules
        assert modules["paraxiallens"] is True


def test_import_path_is_set_when_available(monkeypatch, tmp_path):
    monkeypatch.setenv("DEEPLENS_REPO_PATH", _configured_repo_path(tmp_path))
    adapter = DeepLensAdapter()
    env = adapter.validate_environment()

    if env["available"]:
        assert env["import_path"] is not None
        assert "DeepLens" in env["import_path"]


def test_missing_repo_path_still_works(monkeypatch):
    monkeypatch.delenv("DEEPLENS_REPO_PATH", raising=False)
    adapter = DeepLensAdapter()
    env = adapter.validate_environment()
    assert isinstance(env["available"], bool)
    assert isinstance(env["is_source_checkout"], bool)
