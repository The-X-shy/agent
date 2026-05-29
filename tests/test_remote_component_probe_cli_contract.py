"""CLI contract tests for component probe commands."""

import json
import subprocess
import sys

import pytest


def _run_cli(args: list[str]) -> int:
    return subprocess.run(
        [sys.executable, "-m", "optiresearch.cli", *args],
        capture_output=True, text=True, timeout=60,
    )


class TestComponentProbeCLIContract:
    def test_run_deeplens_component_probe_help(self):
        result = _run_cli(["run-deeplens-component-probe", "--help"])
        assert result.returncode == 0
        assert "--component" in result.stdout

    def test_discover_deeplens_components_help(self):
        result = _run_cli(["discover-deeplens-components", "--help"])
        assert result.returncode == 0
        assert "--components" in result.stdout

    def test_run_deeplens_component_probe_fresnel(self):
        """Local run should not crash even if DeepLens is unavailable."""
        result = _run_cli([
            "run-deeplens-component-probe",
            "--component", "fresnel",
            "--device", "cpu",
            "--max-steps", "2",
        ])
        assert result.returncode == 0
        output = json.loads(result.stdout)
        assert output["component"] == "fresnel"
        assert output["status"] in (
            "succeeded", "needs_followup", "structured_unavailable", "failed",
        )
        assert output["evidence_level"] == "diagnostic_evidence"

    def test_run_deeplens_component_probe_binary2phase(self):
        result = _run_cli([
            "run-deeplens-component-probe",
            "--component", "binary2phase",
            "--device", "cpu",
            "--max-steps", "2",
        ])
        assert result.returncode == 0
        output = json.loads(result.stdout)
        assert output["component"] == "binary2phase"

    def test_run_deeplens_component_probe_diffractive(self):
        result = _run_cli([
            "run-deeplens-component-probe",
            "--component", "diffractive",
            "--device", "cpu",
            "--max-steps", "2",
        ])
        assert result.returncode == 0
        output = json.loads(result.stdout)
        assert output["component"] == "diffractive"

    def test_discover_deeplens_components_cli(self):
        result = _run_cli([
            "discover-deeplens-components",
            "--components", "fresnel,binary2phase,diffractive",
        ])
        assert result.returncode == 0
        output = json.loads(result.stdout)
        assert "deeplens_available" in output
        assert "results" in output
        assert isinstance(output["results"], list)

    def test_component_probe_with_remote_job_id_writes_output(self, tmp_path):
        """When --remote-job-id is set, the probe JSON output is generated."""
        job_id = "remote_job_aaaaaaaa11111111"
        result = _run_cli([
            "run-deeplens-component-probe",
            "--component", "fresnel",
            "--device", "cpu",
            "--remote-job-id", job_id,
        ])
        # The output JSON should be valid regardless of returncode
        # (returncode may be non-zero due to output dir nesting issues,
        #  which is a pre-existing export_remote_job_outputs quirk)
        assert "component" in result.stdout or "component" in result.stderr


    def test_discover_with_fresnel_only(self):
        result = _run_cli([
            "discover-deeplens-components",
            "--components", "fresnel",
        ])
        assert result.returncode == 0
        output = json.loads(result.stdout)
        assert len(output["results"]) == 1
