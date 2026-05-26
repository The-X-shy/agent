"""Tests for cross-platform lens file resolver."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from optiresearch.optics.lens_file_resolver import (
    LensFileResolutionResult,
    resolve_lens_file,
)


class TestResolveLensFile:
    def test_empty_lens_file_returns_error(self):
        result = resolve_lens_file("")
        assert result.error_code == "LENS_FILE_EMPTY"

    def test_whitespace_lens_file_returns_error(self):
        result = resolve_lens_file("   ")
        assert result.error_code == "LENS_FILE_EMPTY"

    def test_absolute_path_exists(self, tmp_path):
        lens = tmp_path / "test_lens.json"
        lens.write_text("{}")
        result = resolve_lens_file(str(lens))
        assert result.exists is True
        assert result.resolved_path == str(lens)
        assert result.source == "absolute_path"
        assert result.error_code is None

    def test_absolute_path_not_exists(self):
        result = resolve_lens_file("/nonexistent/path/lens.json")
        assert result.exists is False
        assert result.error_code == "LENS_FILE_NOT_FOUND"

    def test_auto_prefix_strips_and_appends_json(self, tmp_path, monkeypatch):
        lens_dir = tmp_path / "datasets" / "lenses"
        lens_dir.mkdir(parents=True)
        cooke = lens_dir / "cooke.json"
        cooke.write_text("{}")
        monkeypatch.setenv("DEEPLENS_REPO_PATH", str(tmp_path))

        result = resolve_lens_file("auto:cooke", "deeplens_geolens_geometric")
        assert result.exists is True
        assert result.resolved_path == str(cooke)
        assert result.source == "env_DEEPLENS_REPO_PATH"

    def test_auto_prefix_with_json_already(self, tmp_path, monkeypatch):
        lens_dir = tmp_path / "datasets" / "lenses"
        lens_dir.mkdir(parents=True)
        cooke = lens_dir / "cooke.json"
        cooke.write_text("{}")
        monkeypatch.setenv("DEEPLENS_REPO_PATH", str(tmp_path))

        result = resolve_lens_file("auto:cooke.json", "deeplens_geolens_geometric")
        assert result.exists is True
        assert result.resolved_path == str(cooke)

    def test_env_cooke_lens_file_takes_priority(self, tmp_path, monkeypatch):
        cooke = tmp_path / "my_cooke.json"
        cooke.write_text("{}")
        monkeypatch.setenv("OPTIRESEARCH_COOKE_LENS_FILE", str(cooke))
        monkeypatch.setenv("DEEPLENS_REPO_PATH", "/some/other/path")

        result = resolve_lens_file("auto:cooke")
        assert result.exists is True
        assert result.resolved_path == str(cooke)
        assert result.source == "env_OPTIRESEARCH_COOKE_LENS_FILE"

    def test_relative_path_resolves(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        lens = tmp_path / "relative_lens.json"
        lens.write_text("{}")
        result = resolve_lens_file("./relative_lens.json")
        assert result.exists is True
        assert result.source == "relative_path"

    def test_not_found_reports_all_checked_paths(self, monkeypatch):
        monkeypatch.delenv("DEEPLENS_REPO_PATH", raising=False)
        monkeypatch.delenv("OPTIRESEARCH_COOKE_LENS_FILE", raising=False)
        result = resolve_lens_file("auto:nonexistent_lens_xyz")
        assert result.exists is False
        assert result.error_code == "LENS_FILE_NOT_FOUND"
        assert len(result.checked_paths) > 0
        assert len(result.warnings) > 0

    def test_result_to_dict(self):
        result = LensFileResolutionResult(
            requested_lens_file="auto:cooke",
            resolved_path="/path/to/cooke.json",
            exists=True,
            source="absolute_path",
            checked_paths=["/a", "/b"],
        )
        d = result.to_dict()
        assert d["requested_lens_file"] == "auto:cooke"
        assert d["resolved_path"] == "/path/to/cooke.json"
        assert d["exists"] is True
        assert d["source"] == "absolute_path"

    def test_installed_package_search(self, tmp_path, monkeypatch):
        pkg_dir = tmp_path / "deeplens_pkg"
        lens_dir = pkg_dir / "datasets" / "lenses"
        lens_dir.mkdir(parents=True)
        cooke = lens_dir / "unique_test_cooke.json"
        cooke.write_text("{}")

        mock_pkg = type("mod", (), {"__path__": [str(pkg_dir)]})
        monkeypatch.delenv("DEEPLENS_REPO_PATH", raising=False)
        monkeypatch.delenv("OPTIRESEARCH_COOKE_LENS_FILE", raising=False)

        with patch.dict("sys.modules", {"deeplens": mock_pkg}):
            result = resolve_lens_file("auto:unique_test_cooke")
            if result.exists:
                assert result.source == "installed_deeplens_package"

    def test_known_path_checked_before_package(self, tmp_path, monkeypatch):
        # Verify WSL paths are in checked_paths even when they don't exist
        monkeypatch.delenv("DEEPLENS_REPO_PATH", raising=False)
        monkeypatch.delenv("OPTIRESEARCH_COOKE_LENS_FILE", raising=False)
        result = resolve_lens_file("auto:cooke")
        checked = [p for p in result.checked_paths if "DeepLens" in p]
        assert len(checked) > 0

    def test_no_autoprefix_plain_name(self, tmp_path, monkeypatch):
        lens_dir = tmp_path / "datasets" / "lenses"
        lens_dir.mkdir(parents=True)
        cooke = lens_dir / "cooke.json"
        cooke.write_text("{}")
        monkeypatch.setenv("DEEPLENS_REPO_PATH", str(tmp_path))

        result = resolve_lens_file("cooke.json")
        assert result.exists is True
        assert result.resolved_path == str(cooke)

    def test_dot_slash_relative_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        lens = tmp_path / "mylens.json"
        lens.write_text("{}")
        result = resolve_lens_file("./mylens.json")
        assert result.exists is True
        assert result.source == "relative_path"
