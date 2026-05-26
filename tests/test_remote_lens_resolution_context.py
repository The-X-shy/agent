"""Tests for remote lens resolution context and env var propagation."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from optiresearch.remote.ssh_runner import _build_env_prefix, build_job_command, validate_remote_command
from optiresearch.schemas.remote import RemoteJobSpec, RemoteWorkerSpec


def _make_worker(worker_id="test_wsl", **overrides):
    defaults = {
        "worker_id": worker_id,
        "host": "testhost",
        "port": 22,
        "username": "testuser",
        "ssh_key_path": None,
        "remote_project_dir": "/home/test/project",
        "remote_workspace_dir": "/home/test/project/workspace",
        "python_executable": "/usr/bin/python3",
        "backend_tags": ["wsl", "deeplens", "torch", "remote"],
        "capabilities": {},
    }
    defaults.update(overrides)
    return RemoteWorkerSpec(**defaults)


def _make_diag_job(job_type, **cli_args):
    return RemoteJobSpec(
        job_id="remote_job_abcd1234abcd1234",
        job_type=job_type,
        objective="test diagnostic",
        cli_args=cli_args,
        input_artifacts=[],
        expected_outputs=["result.json"],
        timeout_seconds=600,
        evidence_policy={},
    )


class TestEnvPrefix:
    def test_no_env_vars_returns_empty(self, monkeypatch):
        monkeypatch.delenv("DEEPLENS_REPO_PATH", raising=False)
        monkeypatch.delenv("OPTIRESEARCH_COOKE_LENS_FILE", raising=False)
        assert _build_env_prefix() == ""

    def test_deeplens_repo_path_included(self, monkeypatch):
        monkeypatch.setenv("DEEPLENS_REPO_PATH", "/tmp/deeplens")
        monkeypatch.delenv("OPTIRESEARCH_COOKE_LENS_FILE", raising=False)
        prefix = _build_env_prefix()
        assert "DEEPLENS_REPO_PATH" in prefix
        assert "/tmp/deeplens" in prefix

    def test_both_env_vars_included(self, monkeypatch):
        monkeypatch.setenv("DEEPLENS_REPO_PATH", "/tmp/deeplens")
        monkeypatch.setenv("OPTIRESEARCH_COOKE_LENS_FILE", "/tmp/cooke.json")
        prefix = _build_env_prefix()
        assert "DEEPLENS_REPO_PATH" in prefix
        assert "OPTIRESEARCH_COOKE_LENS_FILE" in prefix
        assert "&&" in prefix


class TestBuildJobCommandDiagnostics:
    def test_trainable_param_inspection_command(self):
        worker = _make_worker()
        job = _make_diag_job("deeplens_trainable_parameter_inspection",
                             lens_file="auto:cooke", device="cpu")
        cmd = build_job_command(worker, job)
        assert "run-deeplens-trainable-parameter-inspection" in cmd
        assert "--lens-file" in cmd
        assert "--remote-job-id" in cmd

    def test_autograd_audit_command(self):
        worker = _make_worker()
        job = _make_diag_job("deeplens_autograd_audit",
                             lens_file="auto:cooke", device="cpu")
        cmd = build_job_command(worker, job)
        assert "run-deeplens-autograd-audit" in cmd
        assert "--lens-file" in cmd

    def test_resolve_lens_file_command(self):
        worker = _make_worker()
        job = _make_diag_job("resolve_lens_file",
                             lens_file="auto:cooke", backend_id="deeplens_geolens_geometric")
        cmd = build_job_command(worker, job)
        assert "resolve-lens-file" in cmd
        assert "--lens-file" in cmd


class TestAllowlistValidation:
    def test_diagnostic_commands_are_allowlisted(self):
        for cmd_name in [
            "run-deeplens-trainable-parameter-inspection",
            "run-deeplens-autograd-audit",
            "run-deeplens-curriculum-probe",
            "run-deeplens-regularized-probe",
            "resolve-lens-file",
        ]:
            from optiresearch.remote.command_allowlist import ALLOWED_CLI_COMMANDS
            assert cmd_name in ALLOWED_CLI_COMMANDS, f"{cmd_name} not allowlisted"

    def test_validate_trainable_param_inspection(self):
        cmd = ["/usr/bin/python3", "-m", "optiresearch.cli",
               "run-deeplens-trainable-parameter-inspection",
               "--lens-file", "auto:cooke", "--device", "cpu",
               "--remote-job-id", "remote_job_abcd1234abcd1234"]
        result = validate_remote_command(cmd)
        assert result["allowed"] is True

    def test_validate_resolve_lens_file(self):
        cmd = ["/usr/bin/python3", "-m", "optiresearch.cli",
               "resolve-lens-file",
               "--lens-file", "auto:cooke",
               "--backend-id", "deeplens_geolens_geometric",
               "--remote-job-id", "remote_job_abcd1234abcd1234"]
        result = validate_remote_command(cmd)
        assert result["allowed"] is True


class TestWorkerKnownLensRoots:
    def test_worker_with_known_lens_roots_capability(self):
        worker = _make_worker(capabilities={"known_lens_roots": ["/mnt/d/DeepLens", "/opt/deeplens"]})
        assert "known_lens_roots" in worker.capabilities
        assert len(worker.capabilities["known_lens_roots"]) == 2

    def test_worker_without_known_lens_roots(self):
        worker = _make_worker()
        assert "known_lens_roots" not in worker.capabilities
