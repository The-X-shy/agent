"""Tests for lens file resolution integration with diagnostic modules."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest


class TestTrainableParameterInspectionResolution:
    def test_lens_file_not_found_produces_structured_unavailable(self, monkeypatch):
        monkeypatch.delenv("DEEPLENS_REPO_PATH", raising=False)
        monkeypatch.delenv("OPTIRESEARCH_COOKE_LENS_FILE", raising=False)
        from optiresearch.runtime.deeplens_trainable_parameter_inspection import (
            inspect_deeplens_trainable_parameters,
        )
        result = inspect_deeplens_trainable_parameters(
            lens_file="auto:nonexistent_lens_xyz", device="cpu"
        )
        assert result["status"] == "unavailable"
        assert result["error_code"] == "LENS_FILE_NOT_FOUND"
        assert result["requested_lens_file"] == "auto:nonexistent_lens_xyz"
        assert result["resolved_lens_file"] is None
        assert len(result["checked_lens_paths"]) > 0

    def test_resolution_metadata_in_result(self, tmp_path, monkeypatch):
        lens_dir = tmp_path / "datasets" / "lenses"
        lens_dir.mkdir(parents=True)
        cooke = lens_dir / "cooke.json"
        cooke.write_text("{}")
        monkeypatch.setenv("DEEPLENS_REPO_PATH", str(tmp_path))
        from optiresearch.runtime.deeplens_trainable_parameter_inspection import (
            inspect_deeplens_trainable_parameters,
        )
        result = inspect_deeplens_trainable_parameters(
            lens_file="auto:cooke", device="cpu"
        )
        assert result["requested_lens_file"] == "auto:cooke"
        assert result["resolved_lens_file"] == str(cooke)
        assert result["lens_resolution_source"] == "env_DEEPLENS_REPO_PATH"
        assert len(result["checked_lens_paths"]) > 0


class TestAutogradAuditResolution:
    def test_lens_file_not_found_sets_diagnosis_and_error(self, monkeypatch):
        monkeypatch.delenv("DEEPLENS_REPO_PATH", raising=False)
        monkeypatch.delenv("OPTIRESEARCH_COOKE_LENS_FILE", raising=False)
        from optiresearch.runtime.deeplens_autograd_audit import run_deeplens_autograd_audit
        result = run_deeplens_autograd_audit(
            lens_file="auto:nonexistent_lens_xyz", device="cpu"
        )
        assert result["status"] == "unavailable"
        assert result["error_code"] == "LENS_FILE_NOT_FOUND"
        assert "lens_file_not_found" in result["diagnosis"]
        assert result["resolved_lens_file"] is None

    def test_resolution_metadata_in_result(self, tmp_path, monkeypatch):
        lens_dir = tmp_path / "datasets" / "lenses"
        lens_dir.mkdir(parents=True)
        cooke = lens_dir / "cooke.json"
        cooke.write_text("{}")
        monkeypatch.setenv("DEEPLENS_REPO_PATH", str(tmp_path))
        from optiresearch.runtime.deeplens_autograd_audit import run_deeplens_autograd_audit
        result = run_deeplens_autograd_audit(
            lens_file="auto:cooke", device="cpu"
        )
        assert result["requested_lens_file"] == "auto:cooke"
        assert result["resolved_lens_file"] == str(cooke)
        assert result["lens_resolution_source"] == "env_DEEPLENS_REPO_PATH"


class TestCurriculumProbeResolution:
    def test_without_lens_file_has_no_resolution_metadata(self):
        from optiresearch.runtime.deeplens_curriculum_probe import run_deeplens_curriculum_probe
        result = run_deeplens_curriculum_probe(max_steps=1, device="cpu")
        assert result["requested_lens_file"] is None
        assert result["resolved_lens_file"] is None

    def test_with_lens_file_resolves(self, tmp_path, monkeypatch):
        lens_dir = tmp_path / "datasets" / "lenses"
        lens_dir.mkdir(parents=True)
        cooke = lens_dir / "cooke.json"
        cooke.write_text("{}")
        monkeypatch.setenv("DEEPLENS_REPO_PATH", str(tmp_path))
        from optiresearch.runtime.deeplens_curriculum_probe import run_deeplens_curriculum_probe
        result = run_deeplens_curriculum_probe(
            max_steps=1, device="cpu", lens_file="auto:cooke"
        )
        assert result["requested_lens_file"] == "auto:cooke"
        assert result["resolved_lens_file"] == str(cooke)
        assert result["lens_resolution_source"] == "env_DEEPLENS_REPO_PATH"


class TestRegularizedProbeResolution:
    def test_without_lens_file_has_no_resolution_metadata(self):
        from optiresearch.runtime.deeplens_regularized_probe import run_deeplens_regularized_probe
        result = run_deeplens_regularized_probe(max_steps=1, device="cpu")
        assert result["requested_lens_file"] is None
        assert result["resolved_lens_file"] is None

    def test_with_lens_file_resolves(self, tmp_path, monkeypatch):
        lens_dir = tmp_path / "datasets" / "lenses"
        lens_dir.mkdir(parents=True)
        cooke = lens_dir / "cooke.json"
        cooke.write_text("{}")
        monkeypatch.setenv("DEEPLENS_REPO_PATH", str(tmp_path))
        from optiresearch.runtime.deeplens_regularized_probe import run_deeplens_regularized_probe
        result = run_deeplens_regularized_probe(
            max_steps=1, device="cpu", lens_file="auto:cooke"
        )
        assert result["requested_lens_file"] == "auto:cooke"
        assert result["resolved_lens_file"] == str(cooke)
        assert result["lens_resolution_source"] == "env_DEEPLENS_REPO_PATH"
